---
type: training-experiment
status: planned
created: 2026-07-22
model: thomas-sounack/BioClinical-ModernBERT-base@c3648aa87af95837c809e6f0c5f85d08160db437 — frozen semantic relation, candidate-conditioned context, selected-action memory, and additive privacy controller
dataset: Three-document ACI structural smoke slice plus the full eligible profile-count target set; final held-out attacker data excluded
result: pending
tags: [ranker, rl, semantic, privacy-head, candidate-context, selected-action-memory, cache-only, aci-contaminated]
companion: ../../docs/specs/RL/ranker-v2-architecture.md
---

# RL-ranker v6 semantic privacy and context vertical slice

## Objective & hypothesis

Validate the selected semantic-ranker architecture through its real local representation and
privacy-pretraining machinery, then exercise behavior cloning, ExIt collection, and hybrid training
under the existing cache-only remote-work boundary. The mechanics hypothesis is that frozen semantic
representations, the profile-relative privacy head, candidate-conditioned document context,
selected-action memory, and the additive controller compose without metadata shortcuts or artifact
identity drift. This smoke does not test promotion, realized privacy, or out-of-corpus generalization.

## Training data

- Frozen environment: `results/ranker_v2/environment/ranker-env.json`.
- Utility assertions: `results/ranker_v2/qa/aci-full.utility`.
- Privacy supervision: eligible rows only from
  `results/ranker_v2/reward/profile-count-targets.json`.
- Structural smoke documents: `aci/D2N001`, `aci/D2N048`, and `aci/D2N002`, carrying respectively
  both assertion families, delivered-only accepted utility, and repeated-decision occurrence
  coverage. Selection is preregistered from the superseded v5 record.
- Privacy-head grouping is by complete `profile_id`; the deterministic split seed is `1729`.
- ACI is development-only encoder-contaminated data. No result from this record may support encoder
  selection or out-of-corpus generalization.

## Training config

- Encoder: `thomas-sounack/BioClinical-ModernBERT-base`, revision
  `c3648aa87af95837c809e6f0c5f85d08160db437`, `trust_remote_code=False`, frozen evaluation mode,
  cache-only model loading.
- Relation arm: `joint-pair`; context mode: `full-candidate-attention`; history mode:
  `selected-cross-attention`; production decision order: `first-occurrence`.
- Privacy smoke seeds: `11`, `29`, `47`; split seed: `1729`; Adam learning rate `0.001`;
  `--max-steps 1`, the smallest positive value, produced finite train/development metrics and
  loadable checkpoints.
- Policy smoke seed: `17`; Adam learning rate `0.0001`; at least two rollouts per document; one
  lambda-zero and one nonzero scheduled episode.
- Counterfactual work follows the frozen preflight budget when its content-addressed manifest is
  available. The manifest and exact budget are currently pending cache-only preflight publication;
  no value is invented for this record.
- Remote-task and context-reader execution is strictly cache-only. A miss must terminate with the
  machine-readable work counts before any external call.

## Selection & operating point

This run is a mechanics slice. It does not select an encoder, architecture arm, lambda menu, or
promotion threshold. Profile-relative count remains an exact shaping target, not a privacy outcome.
The additive controller is checked at lambda zero and at one registered nonzero profile only if the
content-addressed lambda menu exists.

## Evaluation & success criteria

- The three-document representation manifest validates all tensor hashes, source-token counts,
  occurrence coverage, relation coverage, and pinned encoder identity.
- Privacy smoke metrics are finite and each produced checkpoint reloads under its complete contract.
  Structural inspection covers KEEP, fine, coarse, and placeholder modes without recording source
  text or entity values.
- Cache-only `bc`, `exit-collect`, and `train` expose the existing machine-readable miss contract,
  report exact remote-task and context-reader work counts, and launch no remote call.
- A completed real hybrid optimizer step must have nontrivial utility components, finite opposing
  alpha gradients, exact lambda-zero identity, and a loadable semantic checkpoint. Without it this
  record remains `planned` or `running` and names the precise boundary.
- The focused Task 9 suite, repository-configured lint/format checks, and `git diff --check` pass.

## Results

The local mechanics slice ran, but the record remains `planned`: no remote-backed semantic optimizer
step, verified ExIt winner, hybrid utility component, opposing alpha-gradient observation, or policy
checkpoint exists.

- The three-document representation store contains 3 documents, 32 policy decisions, 147
  occurrences, 64 controlled occurrences with 64 covered, 176 actions, and 176 relation entries.
  It contains 5,600 unique document-token rows and 173 tensor files. Build wall time was 16.89
  seconds and peak RSS was 1,549,928 KiB.
- A separate full-environment store was required by the full eligible privacy join. It contains 67
  documents and 3,686 relation entries. Build wall time was 188.75 seconds and peak RSS was
  1,625,788 KiB.
- Privacy pretraining consumed 2,122 eligible level rows from 343 profiles and 663 decisions. All
  three one-step reports are finite and all three checkpoints reload against their exact contracts.
  Wall time was 5.70 seconds and peak RSS was 1,171,644 KiB.
- A four-mode structural inspection had 4 finite predictions and 4 valid intervals. Predicted-count
  ascending order was KEEP, placeholder, fine, coarse; normalized-score ascending order was KEEP,
  fine, coarse, placeholder. KEEP was exact 0 and placeholder exact 1 after normalization.
- Cache-only BC stopped with
  `CACHE_ONLY_MISS phase=initial remote_tasks=3 context_reader_work_items=45` and published no BC
  checkpoint. A separate two-profile, two-rollout cache preflight stopped with
  `CACHE_ONLY_MISS phase=initial remote_tasks=12 context_reader_work_items=180`.
- Real lambda-zero composition passed 32 of 32 tensor-identity checks. Six local structural episodes
  produced 60 selected-privacy comparisons across 7 mode/type/provenance strata. Twelve reverse or
  seeded order replays differed from production by 0 to 6 selected actions, 31 in total.
- Utility, count, entropy, KL, counterfactual, and gradient-norm terms are unavailable because the
  initial utility cache miss stopped execution before ExIt collection and hybrid replay.

## Ablations

No ablation is promoted from this slice. The architecture-fitness harness retains relation,
context, history, shortcut, and order baselines for a later matched run with registered thresholds
and a non-ACI evaluation manifest.

## Cost

The representation and privacy mechanics smoke were local and each completed below ten minutes.
GPU admission showed no KFD processes, but the project environment reported CUDA unavailable, so
both model workloads used CPU. Remote inference and reader execution were not dispatched; cache
misses were recorded instead.

## Risks & caveats

- The full eligible privacy target set can reference relations outside the three-document
  representation slice, so privacy pretraining required the separately pinned full-environment
  representation manifest.
- The remote utility cache and preflight outputs may be absent. That is an expected validation
  boundary, not permission to fabricate artifacts or make a live call.
- The first cache-only model attempt exposed a Transformers adapter-discovery bug that initiated a
  failed DNS/HEAD lookup despite `local_files_only`; no socket messages or remote response were
  recorded. The loader now forwards cache-only state into adapter lookup and its retry remained
  local.
- ACI contamination bars encoder-selection and generalization claims.
- Count targets are shaping supervision and are not evidence of realized privacy against an
  attacker.

## Artifacts

- Code revision at preregistration: `a603f55e74330146bb891c6568efc31d70617dd2`.
- Environment identity: artifact
  `sha256:4cc754a7143252613d2ef0160d7778580621fd973a32e5a0388da510170ddc8a`, file
  `sha256:07f568af1c63d4dff007d95ea58a3540e585e974a3ab26590b927fd6aec42583`.
- Utility identity: artifact
  `sha256:633250a2ecc22bf09df779eaf6e65354bac0a144621c05a5dfcb408c7f5e9b18`, file
  `sha256:276aa0cc6ca2b0994cf791ab23c35e02dad5ddbe4e70a293af5c34242299412f`.
- Profile-target identity: artifact
  `sha256:8c51eced3b5a8638a72eb60b286e3defcc6d4d2304892c2c912ca3247060aa44`, file
  `sha256:a39c3d6a96fd438651f878b1144a1a92a811c9b78181eaec5af553cf2f97e3eb`.
- Smoke representation manifest: `results/ranker_v2/architecture/representation/manifest.json`;
  manifest `sha256:fb655e2ff41a9401a575657f2f1b359e9817f54f0597c0518bdae26ee55be06f`,
  file `sha256:69ff4608d929b366e230e7f3b7456b8e8af341b0fe5110f7ca9cd196c3866c3d`.
- Full representation manifest: `results/ranker_v2/architecture/representation-full/manifest.json`;
  manifest `sha256:17d3c7a5e15179c3354235596e68613f780a1eb9a80549a257dbe61f285edf11`,
  file `sha256:5f396ecee4d04e8eb21795b5923a39c5445754638b2253a313fbd572839ab07d`.
- Privacy split manifest:
  `sha256:9719f8e9e075cc6f1bdb749393456d59863ca47f86f51563747aa19f00ffa687`.
- Privacy metric report:
  `sha256:cb947b7ae3f32d86d020d201ae1bc74a4c3cdc742b3206cfefbe4824498bb77d`.
- Privacy diagnostic manifest:
  `sha256:1730d658b623a2406bc4284e5c31c6654f7876fea1a662ad093e0fa400e7948d`.
- Privacy checkpoint file hashes for seeds 11, 29, and 47 respectively:
  `sha256:4d727b7c76063401b39a8124623dcfb01d91be11c131aa27a78b8a4159c3e96f`,
  `sha256:5f86e2aa524ea0519860bf22648770799f2bcccecd812918587c90b2f5415f64`, and
  `sha256:c26a81eb98b164e43ea8b0c0256899d23ebaf086be1612544190b8720d36ac17`.
- Lambda menu and threshold manifest: `results/ranker_v2/preflight/lambda-menu.json` and
  `results/ranker_v2/preflight/threshold-manifest.json`; hashes pending cache-only preflight.
- Utility cache: `results/ranker_v2/cache/utility-results.jsonl`.
- Policy output directory: `results/ranker_v2/architecture/policy/`; no BC, ExIt, hybrid, or control
  checkpoint was published.
- Predecessor: [RL-ranker v5 interactive count-conditioned policy](2026-07-22-RL-ranker-v5-interactive-count-conditioned.md), superseded before running.

## Sources

- [Ranker v2 architecture](../../docs/specs/RL/ranker-v2-architecture.md).
- [Interactive ranker v2 diagnostics](../../docs/specs/RL/interactive-ranker-v2-diagnostics.md).
- [RL-ranker v5 interactive count-conditioned policy](2026-07-22-RL-ranker-v5-interactive-count-conditioned.md).
