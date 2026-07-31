---
type: paper
node_id: paper:sun2024_rethinking_bradley_terry
title: "Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives"
authors: ["Hao Sun", "Yunyi Shen", "Jean-Francois Ton"]
year: 2024
venue: "arXiv preprint (no published venue stated in v2)"
external_ids:
  arxiv: "2411.04991"
  doi: null
  s2: null
tags: ["reward-modeling", "bradley-terry", "order-consistency", "regression-vs-preference", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives

## Why this paper was surfaced

We went looking for a paper that directly compares *regression on reward differences* against *preference-likelihood training* — the (a) vs (b) fork in our critic design. This is the closest thing that exists: it asks why the Bradley-Terry likelihood is used at all in reward modeling, shows what property the likelihood is actually needed for, and proposes non-BT objectives that satisfy that property. It is the paper that tells us how much the choice of likelihood *matters* in principle, as opposed to which specific tie model to use.

## One-line thesis

The Bradley-Terry likelihood is sufficient but not necessary for reward modeling: downstream optimization only needs *order consistency* (the learned reward must be a monotone transform of the true reward, preserving rankings), so any order-consistent objective — including a plain binary classifier over individual prompt-response pairs — is a valid substitute, and the choice should be made empirically rather than by appeal to BT's probabilistic story.

## Key Results

- **Convergence theory for BT reward models.** They establish a convergence rate for BT reward models implemented as deep networks over embeddings ("BT regression": embeddings → scalar reward, rather than a free parameter per item), giving the first theoretical footing for the standard RLHF practice — which is *not* the classical BT setting, since only a sparse subset of pairs is ever compared.
- **Order consistency is the operative requirement.** Formally, (r̂(x₁,y₁) − r̂(x₂,y₂))(r(x₁,y₁) − r(x₂,y₂)) > 0. If this holds, downstream RL is unaffected by any monotone distortion of reward *values*; therefore calibrated preference probabilities are not needed. BT possesses order consistency — but so do other objectives.
- **A simpler alternative.** They propose an upper-bound objective compatible with off-the-shelf binary classifiers: classify individual prompt-response pairs (marginalizing over the joint comparison probability) and use the logits as rewards. This drops BT's antisymmetric difference structure entirely.
- **Scale of the empirical study.** Over 12,000 experimental setups, 6 base LLMs, 2 datasets, and annotation designs varied along quantity, quality (noise), and pairing choice (same-prompt vs cross-prompt comparisons). The paper's framing is that the BT-vs-alternative question is an empirical one, resolved per annotation regime rather than by theory.

## Relevance to This Project

**It licenses the question and sets its terms.** The paper's core message maps onto our fork cleanly: if all we need from q_U is that it *orders* action pairs correctly (which is exactly what an equivalence critic used for tie-breaking needs — it feeds a decision, not a calibrated utility estimate), then the likelihood family is a free design choice, and the criterion for choosing is not "which is the correct probability model" but "which objective yields order consistency most reliably on our evidence mix". That reframes the Huber-vs-tie-likelihood decision as an empirical one to be settled by a run, not a modelling-principle argument.

**It also marks where our problem is genuinely different, in our favour.** Their alternatives all consume *comparison outcomes* (binary labels). We have measured ΔU *magnitudes* on live pairs — strictly more information than any objective in this paper uses. Regression exploits magnitude; a preference likelihood throws it away and only consumes sign plus a tie flag. That is the one strong argument on the Huber side of the fork, and it is why the hybrid (magnitudes where they are above noise, tie likelihood where they are not) is the option that uses all the evidence we actually have.

**It cautions against over-reading calibration.** If our critic is only ever used to decide equivalence versus difference, the calibration of q_U differences to ΔU units is a means, not an end — and the 0.044 threshold is the only place where absolute scale must be meaningful. That argues for keeping the critic's output in ΔU units (so the threshold has a fixed interpretation) while not over-investing in fitting magnitudes precisely.

## Design question it bears on

(a) Huber difference regression vs (b) tie-aware preference likelihood: this paper says neither is privileged a priori — order consistency is the requirement, and multiple objective families achieve it — so the decision should be made on which objective handles our specific evidence strata (39% tie-class mass, measured magnitudes on the rest) without degenerating. It is also the anchor for the sub-question "does regression-on-differences vs preference-likelihood have a direct comparison paper": partially, and at the level of objective *family* rather than of regression-on-measured-gaps.

## Caveats

- We could not extract the per-setup numeric results from the HTML render; the verified content is the theoretical framing (order consistency, convergence rate, classification alternative) and the study's scale, not a specific win/loss margin between BT and the classifier. Any claim of the form "regression beats BT by X" should not be attributed to this page without reading the paper's result tables.
- The alternative they benchmark is a *pointwise binary classifier*, not regression on continuous measured utility gaps — so this is an adjacent comparison, not our exact arm.
- No tie handling at all: ties are outside this paper's scope. It informs the (a)-vs-(b) framing, not the tie-treatment question.
- Preprint, no stated venue in v2.

## Sources

- [arXiv 2411.04991](https://arxiv.org/abs/2411.04991) — Sun, Shen, Ton. *Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives.*
- Tie-aware objectives this paper does not cover: [chen2025_dpo_ties.md](chen2025_dpo_ties.md) ([arXiv 2409.17431](https://arxiv.org/abs/2409.17431)), [liu2024_reward_learning_ties.md](liu2024_reward_learning_ties.md) ([arXiv 2410.05328](https://arxiv.org/abs/2410.05328)).
