---
type: experiment
node_id: exp:epsilon-zero-lexicographic-gate
status: planned
verdict: pending
confidence: pending
created: 2026-08-04
updated: 2026-08-04
tags: [rl, ranker-v2, lexicographic, exact-ties, cache-only]
companion: ../../docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md
---

# Epsilon-zero lexicographic gate — does the exact-utility tie set contain free count gain?

## Hypothesis

Among complete cached vectors with maximum pinned document utility, deterministic
selection of the highest frozen profile-count score yields positive document-macro
count gain outside the four controller-campaign documents with exactly zero measured
utility loss, provided every primary document has complete coverage of the frozen
five-anchor standardized candidate slate.

## Forbidden substitutions

- no 0.044 threshold;
- no structural tie labels;
- no utility-tower logits in admissibility;
- no alpha/gain calibration;
- no per-decision composition;
- no reward calls during the gate.
- no zero-imputation for missing standardized candidates.

## Why this experiment

Five controller rounds (v11–v15, then the count-to-gain and authority-interval spikes) tried to make an additive `alpha * g(lambda) * p_hat` shift own reward ties. The [root-cause report](../../docs/research/tie-ownership-root-cause-and-solution-space.md) established that the composition itself is the defect: where the primary gradient vanishes, no gradient-space composition can localise the decision, and the prescribed operator in the literature we had already registered is a **filter** over the primary-optimal set, not a bonus. This gate executes that operator directly on already-measured complete trajectories, so its outcome cannot be defeated by tower-logit scale, dead gain features, or weak `alpha` gradients — none of them are inputs.

## Design

For document `d`, `C_d` is the frozen five-anchor standardized slate (`build_anchor_trajectories`) joined to validated cached results; `U_d` is pinned QA-v2 document utility; `P_d` is the frozen profile-relative count score.

- `F_d^0 = {v in C_d : U_d(v) = max U_d}` under an exact `Decimal` weighted-utility key reconstructed from pinned component scores and weights — floating-point representation tolerance is not semantic slack.
- `epsilon_zero_lexicographic` selects `argmax_{v in F_d^0} P_d(v)`, ties broken by stable vector key.
- `utility_only_bc_nearest` selects, inside the same `F_d^0`, the vector with minimum Hamming distance to the lambda-independent BC teacher, then stable vector key — privacy-blind.
- `G_d = P_d(lex) - P_d(utility_only) >= 0`.
- Archived additive checkpoints (`detached-s47`, `coupled-s47`) are replayed greedily at lambda-zero and lambda-max as **report-only** comparators.

Documents are the statistical unit. Support status is assigned before any gain is computed: a document with a missing standardized anchor, or with no count contrast inside its complete slate, is `G_d: null` — never `G_d = 0`.

### Populations

- **Primary:** every retained document except `aci/D2N005`, `aci/D2N027`, `aci/D2N031`, `aci/D2N063`.
- **Campaign diagnostic:** those four, reported separately.

### Decision rule (evaluated in order)

1. `INVALID` — illegal/incomplete selected vector, a lexicographic selection below the document utility maximum, an artifact-pin difference, conflicting cache results, or a non-reproducible hash.
2. `INSUFFICIENT-CANDIDATE-BREADTH` — any primary document is not support-complete.
3. `NO-OBSERVED-EXACT-OPPORTUNITY` — every support-complete primary document has `G_d = 0`.
4. `INSUFFICIENT-PRIMARY-SUPPORT` — some `G_d > 0`, but the one-sided 95% document-bootstrap lower bound for mean `G_d` (10,000 resamples, seed 20260804) is zero.
5. `ADOPT-EXACT-LEXICOGRAPHIC-COMPOSITION` — all primary documents support-complete, selector valid, at least one primary document changes vector, and the bootstrap lower bound is positive.

## Frozen inputs

| artifact | sha256 |
|---|---|
| `results/ranker_v2/environment/ranker-env.json` | `07f568af1c63d4dff007d95ea58a3540e585e974a3ab26590b927fd6aec42583` |
| `results/ranker_v2/qa/aci-full.utility` | `276aa0cc6ca2b0994cf791ab23c35e02dad5ddbe4e70a293af5c34242299412f` |
| `results/ranker_v2/reward/profile-count-targets.json` | `a39c3d6a96fd438651f878b1144a1a92a811c9b78181eaec5af553cf2f97e3eb` |
| `results/ranker_v2/cache/utility-results.jsonl` | `a787a54ac70349da36e91f408386460ca0e26457318a69c9a2df8f772014649d` |

Comparator checkpoints: `results/ranker_v2/architecture/count_to_gain/detached-s47.pt`, `.../coupled-s47.pt` (hashes recorded in the result artifact).

## Evaluation & success criteria

Cache-only, CPU. The run must report zero remote tasks and zero reader work items. Two runs must produce byte-identical canonical JSON. Per-document readouts, aggregate readouts, and the closed verdict enum are specified in §3.4–3.5 of the [companion plan](../../docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md).

## What a pass does not license

- No claim that the current tower can identify `F_d^0` on unseen documents.
- No privacy claim: the count score is a frozen diagnostic proxy. Attacker-measured realized privacy on paired `utility_only` / `lexicographic` outputs is a separate, explicitly approved run.
- No nonzero utility budget, and no more than two lambda regimes at `epsilon = 0`.
- No claim that the frozen five-anchor slate contains the global optimum over all legal vectors.

## Real-data preflight

Executed 2026-08-04 on `aci/D2N005` (campaign) plus the first two sorted support-complete non-campaign documents, selected from `candidate_support_status` before any gain was computed. Cache-only, no reward calls, no comparator checkpoints. Retained documents 63; validated candidates 327 of the 330 rows belonging to these three documents (3 excluded as `reader_refresh`, 0 pin mismatches, 0 incomplete, 0 illegal, 0 parity failures).

| document | population | slate | cached | exact-tie set | `G_d` | lexicographic vector | utility-only vector |
|---|---|---|---|---|---|---|---|
| `aci/D2N001` | primary | 4 | 4 | 2 | 0.0 | `sha256:0f1a3bee…` | `sha256:0f1a3bee…` |
| `aci/D2N002` | primary | 4 | 4 | 1 | 0.0 | `sha256:e48dd53d…` | `sha256:e48dd53d…` |
| `aci/D2N005` | campaign | 4 | 4 | 2 | 0.0 | `sha256:62963cbf…` | `sha256:62963cbf…` |

Manual verification: both selectors' vectors are cache rows (each carries a `result_hash`); every standardized anchor is present for all three; recomputed float utility matches the stored value on every retained row (`utility_parity_excluded = 0`); no structural dependency label, tower logit, or `alpha` is read anywhere in the gate; the lexicographic pick equals the maximum-count member of the exact-optimal set in every case. The `unsupported` branch could not be exercised on real data — **no** primary document is unsupported — so it is covered by fixtures only.

**Two observations that bear on interpretation, recorded before the full run.** First, the five anchors deduplicate to **four** vectors per document, and their exact-utility argmax sets have size 1–2, so the standardized population offers very few tied pairs to choose between. Second, where a tie does exist the privacy-blind BC-nearest comparator already selects the maximum-count member, giving `G_d = 0`. The non-adjudicating expanded-cache diagnostic over the adaptively sampled pool (65 / 76 / 186 vectors) reports positive gain on all three documents (0.058 / 0.059 / 0.150) — evidence that the operator finds opportunity in a richer candidate pool, and that a zero corpus result would be a statement about the coarse standardized slate rather than about exact ties as such.

## Results

_pending_

## Sources

- [skalse2022_lexicographic_morl](../papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) — optimise lower-priority objectives only inside the primary's optimal set; exact ties consume no primary slack.
- [wray2015_lexicographic_mdp_slack](../papers/wray2015_lexicographic_mdp_slack.md) ([DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647)) — lexicographic policies can lie outside the linear-scalarisation policy set, so a failed `alpha` search is not evidence for more tuning.
- [hayes2022_practical_guide_morl](../papers/hayes2022_practical_guide_morl.md) ([arXiv 2103.09568](https://arxiv.org/abs/2103.09568)) — scalarisation is a substantive preference model, not a neutral default.
- [tercan2024_thresholded_lexicographic](../papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493)) — global thresholded lexicographic methods are unsafe; this gate uses no threshold at all.
- [vamplew2024_value_function_interference](../papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) — random greedy tie-breaking aggravates value-function interference; both selectors here are fully deterministic.
- Predecessor records: [v14 evidence tie ownership](2026-07-31-RL-ranker-v14-evidence-tie-ownership.md), [v15 equivalence critic](2026-07-31-RL-ranker-v15-equivalence-critic.md), [Gate 1 representation](2026-08-01-RL-ranker-gate1-representation.md), [count-to-gain coupling](2026-08-03-count-to-gain-coupling.md).
