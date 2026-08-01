---
type: paper
node_id: paper:kemertas2021_robust_bisimulation_metric
title: "Towards Robust Bisimulation Metric Learning"
authors: ["Mete Kemertas", "Tristan Aumentado-Armstrong"]
year: 2021
venue: "NeurIPS 2021"
external_ids:
  arxiv: "2110.14096"
  doi: null
  s2: null
tags: ["bisimulation-metric", "behavioral-equivalence", "representation-collapse", "sparse-reward", "equivalence-generalization"]
added: 2026-08-01T00:00:00Z
---

# Towards Robust Bisimulation Metric Learning

## Why this paper was surfaced

Bisimulation metrics are the one part of the RL literature that does exactly what our failed equivalence critic was trying to do: learn a *metric* whose small values mean "behaviorally equivalent", and evaluate it on states never seen in training. Before we retry the critic with better data we needed to know how that line of work behaves when the reward signal is uninformative over large regions — which is our situation by construction, since 39% of measured decisions sit at or below the reader floor. This paper is the one that diagnoses that regime rather than assuming it away.

## One-line thesis

Learned bisimulation metrics degenerate when the reward signal is sparse or flat — the embedding's dependence on reward becomes unstable and the representation collapses — so the "learn a behavioral-equivalence metric that generalizes" program needs explicit norm constraints, dynamics regularization, and intrinsic reward to survive, and even then only under stated conditions.

## Key Results

- **Generalized approximation bounds.** Value-function approximation bounds for on-policy bisimulation metrics are extended from the optimal-policy / exact-dynamics case to *non-optimal policies and approximate learned dynamics* — the realistic case, and the one where the guarantee gets weak.
- **Named embedding pathologies.** The theory is used to identify concrete failure modes in practical bisimulation-metric learners (the DBC family): an underconstrained dynamics model, and an unstable dependence of the embedding norm on the reward signal. Both bite hardest **in sparse-reward environments**, where the reward term contributes almost nothing to the metric and the representation collapses toward a constant.
- **Remedies are constraints, not more data.** The fixes are a norm constraint on the representation space plus intrinsic reward and latent-space regularization. Framed our way: when the primary signal cannot separate states, they inject a *different* signal rather than trying harder to fit the flat one.
- **Recovered capability.** With those remedies the method solves continuous-control tasks with observational distractors and sparse rewards where prior bisimulation approaches fail.

## Relevance to This Project

**This is the closest existing thing to "learn a predictor of behavioral equivalence that generalizes to unseen instances", and it is also the paper that explains why ours collapsed.** A bisimulation metric says two states are equivalent iff they have the same reward and equivalent transition distributions. Transposed to our menus: two generalization levels are equivalent iff the reader scores them identically. Our reward field over the equivalence-relevant region is not merely sparse, it is *flat by design* — that is the whole finding of the tie census. The paper's diagnosis is that a flat reward term makes the metric's reward-driven component vanish, so what the embedding actually encodes is whatever the dynamics/feature term happens to encode, and the norm drifts. That maps directly onto v15: the critic's confident region was a four-document neighborhood of the training features, not a region where anything about equivalence was actually learned.

**The prescription transfers better than the method.** Their answer to a flat primary signal is not a bigger predictor — it is (i) constrain the representation so it cannot collapse, and (ii) supply a second, non-degenerate signal. Our analogues are, respectively, the derivable structural equivalence from the QA artifact's assertion→decision dependency links (a hard constraint on which decisions *cannot* move utility, available for free on every corpus document, no probing), and the privacy score, which is already the second channel we route through the tie sets. That reframes the retry: not "train the critic harder on the same 10,206 rows from four documents", but "give the critic label mass that is document-diverse and partly free".

**Honest limit — the reward here is the environment's, ours is an instrument's.** Bisimulation is defined against the MDP's own reward and transition kernel, quantities the agent's model can in principle represent. Our equivalence is defined against an external measurement instrument (the reader scoring QA probes) whose output is not a function of any state we model, is noisy at a measured 0.044 resolution, and — decisively — does not exist at deployment at all. So the bisimulation guarantee "small metric ⇒ small value difference" has no deployment-time counterpart for us: there is no reward to be close to. What survives is the diagnostic, not the theorem.

## Design question it bears on

Is the equivalence-critic retry worth running, and what has to change? This paper says the failure of a learned equivalence metric under a flat reward field is expected and structural, and that the fixes are representation constraints plus an additional signal — not more capacity and not more epochs. It supports one specific retry (document-diverse, partly structurally-derived labels, with an explicit constraint on the accept region) and argues against a straight rerun with richer features.

## Caveats

- Their setting is pixel-based continuous control with distractors; the collapse mechanism is argued through the embedding norm, which our frozen-trunk linear probes do not have in the same form. The mechanism is suggestive for us, not proven.
- The remedies (intrinsic reward, latent regularization) are tied to learning a dynamics model, which we do not have — our "transitions" are a lattice of text edits.
- Bisimulation collapses states that are equivalent *including* on reward; it gives no account of choosing among equivalents by a secondary criterion. That routing remains the unpublished part, as already noted on [baram2021_action_redundancy.md](baram2021_action_redundancy.md).

## Sources

- [arXiv 2110.14096](https://arxiv.org/abs/2110.14096)
- [NeurIPS 2021 proceedings](https://proceedings.neurips.cc/paper/2021/hash/256bf8e6923a52fda8ddf7dc050a1148-Abstract.html)
- Related in this wiki: [vanderpol2020_mdp_homomorphic_networks.md](vanderpol2020_mdp_homomorphic_networks.md) ([arXiv 2006.16908](https://arxiv.org/abs/2006.16908)) — the enforced-structure alternative; [damour2020_underspecification_ml.md](damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)) — what fills the gap when the objective is indifferent.
