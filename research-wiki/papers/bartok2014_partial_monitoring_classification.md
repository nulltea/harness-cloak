---
type: paper
node_id: paper:bartok2014_partial_monitoring_classification
title: "Partial Monitoring—Classification, Regret Bounds, and Algorithms"
authors: ["Gábor Bartók", "Dean P. Foster", "Dávid Pál", "Alexander Rakhlin", "Csaba Szepesvári"]
year: 2014
venue: "Mathematics of Operations Research 39(4):967-997"
external_ids:
  arxiv: null
  doi: "10.1287/moor.2014.0663"
  s2: null
tags: ["partial-monitoring", "identifiability", "duplicate-actions", "observability", "measurement-resolution", "minimax-regret"]
added: 2026-08-01T00:00:00Z
---

# Partial Monitoring—Classification, Regret Bounds, and Algorithms

## Why this paper was surfaced

We have a decision problem in which the feedback channel — a reader scoring QA probes — provably cannot separate certain actions, and a policy that must nonetheless choose among them. We wanted the formal framework where "which action distinctions are learnable from *this* feedback structure" is the central object rather than a nuisance. Partial monitoring is that framework, and this paper is its complete classification theorem.

## One-line thesis

In a game where the learner's feedback is a known but incomplete function of action and outcome, the achievable regret is determined entirely by the *structure of the loss and feedback matrices* — every finite game falls into exactly one of four classes (trivial 0, easy Θ(T^{1/2}), hard Θ(T^{2/3}), hopeless Ω(T)) — and the class is computed from the declared structure, never estimated from data.

## Key Results

- **Complete classification.** Minimax regret is `0` when the game has no neighbouring actions; `Θ(T^{1/2})` when it is locally observable and has neighbouring actions; `Θ(T^{2/3})` when globally but not locally observable; `Ω(T)` (hopeless) otherwise. There is nothing in between — the classification is exhaustive up to constants and log factors.
- **Duplicate actions are collapsed by definition, not learned.** Two actions are *duplicates* iff their loss vectors are identical across all outcomes (`ℓ_a = ℓ_b`). Standard preprocessing removes duplicates and dominated/degenerate actions before the game is classified: identical-loss actions form an equivalence class that the analysis never has to distinguish, because no amount of feedback could distinguish them and no regret is incurred by choosing arbitrarily among them.
- **Observability is a decomposition condition on the *declared* feedback.** Global observability: for every pair of neighbouring actions there exists `f` with `ℒ(a,x) − ℒ(b,x) = Σ_c f(c, Φ(c,x))` — the loss difference is reconstructible from the signals of *some* actions. Local observability additionally requires `f` to vanish outside the neighbourhood, so the difference is estimable from the two actions' own signals. Both are checkable properties of the matrices, decided offline.
- **Hopelessness is an in-principle statement.** In the hopeless class, some loss difference is not a function of any feedback the learner can ever obtain; no algorithm attains sublinear regret. The obstruction is identifiability, not sample size, so more data does not help.

## Relevance to This Project

**It supplies the vocabulary our tie problem has been missing, and it puts our exact-tie stratum on the right side of a formal line.** Our menus are a decision problem with a declared feedback structure: the utility channel is the reader's score over QA probes, and the QA artifact literally declares, per assertion, which decisions can influence it (`policy_dependency_decision_ids`). Two generalization levels of a span that no assertion depends on have *identical loss vectors under that channel* — they are duplicate actions in this paper's precise sense. The framework's treatment of duplicates is the answer we were groping toward: duplicates are identified from the declared structure and collapsed before learning starts. That is a derivation, not a prediction, and it is exactly what 13% of our 652 corpus decisions admit.

**It also names what our 0.044 floor is.** Below the reader's measured resolution, the utility difference between two actions is not a function of anything the feedback can return — the same obstruction that defines the hopeless class, localized to a sub-band rather than to a whole game. The correct move under this framework is not to learn the sub-floor ordering (it is unidentifiable in principle, so no estimator converges) but to *declare* the sub-band a single equivalence class and let a different, observable channel — our privacy score — order inside it. In partial-monitoring terms: we choose the game's action set so that the unidentifiable distinctions are removed, then play a game that is identifiable.

**The honest gap.** Partial monitoring assumes the loss and feedback matrices are *known*. Ours are known only where the QA artifact exists — on corpus documents, at training time. On a real user document there are no QA probes, so neither the loss matrix nor the feedback structure is available and the classification cannot be run. The framework therefore justifies the training-time derivation and the by-policy collapse of the sub-floor band; it does not by itself get either to deployment. That last hop is the part with no paper behind it.

## Design question it bears on

Two of them. (1) Should sub-floor utility differences be *learned* or *declared equivalent*? This says declared: sub-resolution distinctions are unidentifiable in principle, and a framework built around exactly that question responds by removing the distinctions from the action set. (2) Is structural derivation of equivalence a legitimate substitute for measurement? Yes — duplicate elimination from the declared loss structure is the standard, mandatory first step here, and it is free.

## Caveats

- Finite action/outcome games with known matrices; our action menus are generated per document from a lattice and the loss structure is only partially declared.
- Regret is measured against the best fixed action in hindsight under a single loss. Our problem is lexicographic — a primary loss with a secondary criterion inside its indifference sets — which this framework does not model. That composition is the same unpublished piece flagged on [skalse2022_lexicographic_morl.md](skalse2022_lexicographic_morl.md).
- "Hopeless" here means a *global* linear-regret verdict. Our situation is milder and more benign: the unidentifiable distinctions carry zero primary loss by construction, so arbitrary choice among them costs no utility — the cost lands entirely on the secondary (privacy) objective, which is outside the model.

## Sources

- [Mathematics of Operations Research 39(4):967-997, DOI 10.1287/moor.2014.0663](https://doi.org/10.1287/moor.2014.0663)
- Definitions of duplicate actions, neighbours, and local/global observability, plus the classification theorem, restated in [Lattimore & Szepesvári, "An Information-Theoretic Approach to Minimax Regret in Partial Monitoring"](https://arxiv.org/abs/1902.00470) ([arXiv 1902.00470](https://arxiv.org/abs/1902.00470)).
- Related in this wiki: [katzsamuels2019_true_sample_complexity_good_arms.md](katzsamuels2019_true_sample_complexity_good_arms.md) ([arXiv 1906.06594](https://arxiv.org/abs/1906.06594)) — verifiable vs unverifiable identification; [russo2018_satisficing_bandit.md](russo2018_satisficing_bandit.md) ([arXiv 1803.02855](https://arxiv.org/abs/1803.02855)) — what to target once exact optimality is dropped.
