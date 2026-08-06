---
type: training-experiment
status: running
created: 2026-08-04
model: semantic-v1 ranker initialized from thomas-sounack/BioClinical-ModernBERT-base@c3648aa87af95837c809e6f0c5f85d08160db437 frozen representations
dataset: 63 signal-bearing ACI documents from the migrated QA-builder-v2 full-v16 utility artifact; 640 in-scope policy decisions after count/scope filtering
result: pending
tags: [rl, ranker-v2, full-corpus, qa-builder-v2, direct-count, evidence-gain, detached-count]
companion: ../../docs/specs/RL/interactive-ranker-v2.md
---

# RL-ranker v7 full QA-v2 detached-controller run

## Objective & hypothesis

Run the newest retained ranker-v2 trainer configuration on the complete migrated QA-builder-v2
artifact rather than the four-document controller-development subset. The primary mechanics
hypothesis is that fresh full-corpus behavior cloning and utility-only ExIt produce a valid warm
start, after which two balanced four-profile Latin cycles complete without artifact drift,
lambda-zero identity failures, non-finite losses, or trainer crashes.

This is the first full-corpus training execution of the current stack. One development seed is the
smallest run that establishes feasibility and runtime. It does not support seed-stability,
held-out-document generalization, or matched-realized-privacy claims.

## Training data

- Environment: `results/ranker_v2/environment/ranker-env.json`, environment hash
  `sha256:4cc754a7143252613d2ef0160d7778580621fd973a32e5a0388da510170ddc8a`.
- Utility: `results/ranker_v2/qa/aci-full.utility`, migrated from
  `results/qa_v2_aci_full_v16/aci_full.utility`; artifact hash
  `sha256:633250a2ecc22bf09df779eaf6e65354bac0a144621c05a5dfcb408c7f5e9b18`.
- Assertions: 1,357 across 67 source documents. The trainer's frozen signal gate drops
  `aci/D2N048`, `aci/D2N054`, `aci/D2N055`, and `aci/D2N060`, leaving 63 documents.
- Policy scope: 640 trainable decisions after 53 out-of-scope or count-uncovered decisions are
  demoted to fixed KEEP.
- Exact count targets: `results/ranker_v2/reward/profile-count-targets.json`, artifact hash
  `sha256:8c51eced3b5a8638a72eb60b286e3defcc6d4d2304892c2c912ca3247060aa44`.
- Frozen representations: `results/ranker_v2/architecture/representation-full/manifest.json`,
  manifest hash `sha256:17d3c7a5e15179c3354235596e68613f780a1eb9a80549a257dbe61f285edf11`.
- Seed: 47. ACI remains development-only and encoder-contaminated.

## Training config

The run rebuilds every data-dependent warm-start stage on all 63 retained documents:

1. Behavior cloning: 3 epochs, learning rate `1e-4`, lambda zero.
2. Utility-only ExIt: 4 rollouts per document, strict improvement over the BC reference.
3. Hybrid RL: 8 epochs, 8 rollouts per document/profile group, four-profile frozen lambda menu,
   learning rate `1e-4`, `beta=0.01`, and `eta=0.01`. The first full-corpus feasibility pass uses
   the newest trainer's default `--skip-lambda-zero-control`; the required control is a separate
   follow-up only if the conditional pass completes and remains worth evaluating.

The retained controller configuration is direct count, switch-calibrated global alpha, no gap
scaling, utility-logit softcap 25, profile-sensitivity coefficient 0.1, evidence gain with an
online tie-margin hinge, and `count_to_gain=detached`. The rejected count-to-gain coupling arm is
not rerun. Counterfactual coverage is degeneracy-aware; KL is forward and collapse-triggered;
synchronous profile evaluation uses 16 samples; rollouts are batched; remote and reader worker
counts are 6/6. All heavy stages use `--device cuda` in the host `.venv` and execute serially.

## Selection & operating point

No model or hyperparameter selection occurs in this run. The frozen lambda values are
`[0.0, 0.4774907574221311, 1.0499663529479142, 1.5270084986176522]`. The run uses seed 47 only;
additional seeds require this feasibility run to complete and are separate executions.

## Evaluation & success criteria

- Focused semantic and interactive trainer tests pass before launch.
- A one-document cache-only BC preflight and a smallest real hybrid smoke complete before the full
  run, confirming artifact pins, GPU placement, and the expected cache behavior.
- Fresh full-corpus BC and ExIt artifacts contain all 63 retained documents.
- Eight conditional epochs complete with finite losses, exact lambda-zero identity, and a loadable
  checkpoint. The separate lambda-zero control remains required before promotion claims.
- Epoch reports contain all four profiles with balanced exposure over both Latin cycles and record
  utility, count, entropy, KL, counterfactual, tie, gain, and synchronous-profile diagnostics.
- Behavioral responsiveness, lambda-zero utility non-inferiority, frontier regret, and greedy
  profile separation are reported as outcomes; failure is not calibrated away.

## Results

Running. Pre-launch validation:

- Focused trainer verification passed: `167 passed` from
  `PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
  src/cloak/tests/test_semantic_ranker.py src/cloak/tests/test_interactive_ranker.py`.
- The slice-matched one-document preflight completed BC, ExIt, and one exact-stack hybrid epoch.
  Hybrid wall time was 2.7 minutes with finite loss, live utility/count/counterfactual/tie gradient
  families, and zero lambda-identity failures. The full smoke wall time was 4.0 minutes.
- The smoke added 13 utility-cache rows, with only 7 cache hits and 103 transport calls in its
  hybrid epoch. This confirms that fresh full-corpus BC will initially exercise the remote reward
  path rather than replaying the controller-development cache.
- GPU telemetry observed 95% peak utilization and 22% peak VRAM (about 14.1 GiB of 64 GiB). Six
  trainer workers match the six llama-server slots. The final full launch passed the standardized
  performance gate 6/6.

## Ablations

None. This is a single retained configuration, not a controller search.

## Cost

Local-only remote generation through the existing llama-swap service and the shared deterministic
LLM cache. One AMD Strix Halo iGPU process at a time. The dedicated utility cache starts as a
byte-identical copy of the latest detached-controller cache and grows only inside this run.
Measured cold-cache hybrid throughput was 2.7 minutes for one small document-group; previous
warm-cache four-document runs were 5--7 minutes per epoch. The conditional full-corpus estimate is
therefore 12--18 hours with a conservative cold-path upper estimate near 23 hours, plus BC/ExIt.

## Risks & caveats

- The utility cache is uneven: all 67 documents are represented, but many non-development documents
  have only six cached action vectors. ExIt and RL will therefore generate new local remote-model
  work.
- The current working tree contains uncommitted trainer changes. Before launch, the exact source
  file hashes and working-tree diff hash must be recorded; the checkpoint's commit-only
  `code_revision` field is insufficient by itself.
- Full-corpus runtime may be long. The run is launched only after a measured smoke confirms GPU
  use, batching, serial execution, and a defensible wall-time estimate.
- Count score is shaping supervision, not realized privacy. Promotion still requires held-out
  attacker evaluation at matched realized privacy.

## Artifacts

- Output root: `results/ranker_v2/training/full-qa-v2-detached-s47/`.
- Utility cache: `results/ranker_v2/training/full-qa-v2-detached-s47/utility-cache.jsonl`.
- BC checkpoint: `results/ranker_v2/training/full-qa-v2-detached-s47/bc.pt`.
- ExIt winners: `results/ranker_v2/training/full-qa-v2-detached-s47/exit-winners.json`.
- Conditional checkpoint: `results/ranker_v2/training/full-qa-v2-detached-s47/conditional.pt`.
- KL reference: `results/ranker_v2/training/full-qa-v2-detached-s47/kl-reference.pt`.
- Lambda-zero control: `results/ranker_v2/training/full-qa-v2-detached-s47/lambda-zero-control.pt`.
- Epoch reports: `results/ranker_v2/training/full-qa-v2-detached-s47/epochs.jsonl`.
- Logs and launch commands: `results/ranker_v2/training/full-qa-v2-detached-s47/`.

## Sources

- [Interactive ranker v2](../../docs/specs/RL/interactive-ranker-v2.md).
- [Ranker v2 architecture](../../docs/specs/RL/ranker-v2-architecture.md).
- [Ties by design](../../docs/specs/RL/ties-by-design.md).
- [Count-to-gain coupling experiment](../experiments/2026-08-03-count-to-gain-coupling.md).
