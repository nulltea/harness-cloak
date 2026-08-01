---
type: paper
node_id: paper:vanderpol2020_mdp_homomorphic_networks
title: "MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning"
authors: ["Elise van der Pol", "Daniel E. Worrall", "Herke van Hoof", "Frans A. Oliehoek", "Max Welling"]
year: 2020
venue: "NeurIPS 2020"
external_ids:
  arxiv: "2006.16908"
  doi: null
  s2: null
tags: ["mdp-homomorphism", "equivalence-classes", "equivariance", "state-action-abstraction", "generalization-to-unseen-states"]
added: 2026-08-01T00:00:00Z
---

# MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning

## Why this paper was surfaced

The equivalence-critic screening (RL-ranker v15) asked a supervised head to *predict* which action pairs are utility-equivalent on an unseen document, and it failed to transfer (held-out gate AUC 0.60). Before iterating on the predictor we needed the literature's actual position on the question "how does an equivalence relation over actions generalize to states you have never visited?". This is the canonical answer from the MDP-homomorphism line, and its answer is not "predict it".

## One-line thesis

If the state-action equivalence is a *group symmetry of the MDP that you can write down*, you do not learn it and you do not predict it per state: you enforce it as an equivariance constraint inside the policy/value network, so equivalent state-action pairs are mapped to identical outputs by construction, on every state including ones never seen in training.

## Key Results

- **Equivalence enforced structurally, not statistically.** The paper builds MDP homomorphic MLPs and CNNs whose layers are equivariant to a declared group (reflections, rotations) acting jointly on states and actions. The equivalence holds *identically* on held-out states — it is a property of the function class, not of the fitted parameters, so there is no held-out AUC to fail.
- **Numerical construction of equivariant layers.** The practical contribution is a procedure for solving the equivariance constraint numerically (symmetrizing a basis) rather than deriving weight-sharing patterns by hand, which is what made structural equivalence cheap enough to use in deep RL.
- **The benefit is solution-space reduction, i.e. sample efficiency.** Equivariant policy/value networks converge faster than unstructured baselines on CartPole, a grid world, and Pong. The gain is that the network cannot represent a policy that treats equivalent pairs differently.
- **The precondition is a declared group.** The whole apparatus requires the symmetry to be *given* as an invertible group action known in advance. The paper does not discover symmetries from data; that is explicitly out of scope.

## Relevance to This Project

**It says plainly what class of problem "predict the equivalence class" is, and our problem is not in it.** The homomorphism literature achieves unseen-state generalization by moving the equivalence out of the learned function and into the hypothesis class. That works exactly when the equivalence is (a) known a priori and (b) a group action — closed, invertible, composable. Ours is neither. "Lisbon → Portugal preserves every probed answer" is not a group orbit: it is not invertible (you cannot un-generalize), it does not compose into a symmetry of the document, and it is contingent on which assertions the QA probes happen to ask. The one legitimate borrowing is the *design pattern*: wherever we can name a rule that makes two actions equivalent independently of measurement, we should hard-code it as a constraint (canonicalization at inference, or a shared output), not train a head to rediscover it.

**It also sharpens the honest limit of the transfer.** Our equivalence is defined by an external measurement instrument — a reader scoring QA probes — not by the environment's transition kernel. A homomorphism is a statement about dynamics: `P` and `R` commute with the map. Our reader is not part of any dynamics we control or model; the "reward" it emits is an opaque function of a text artifact. So the theorem "equivariant network ⇒ equivalent pairs get equal value on all states" has no analogue for us unless we can express the reader's indifference as a structural map, which is precisely what the QA artifact's `policy_dependency_decision_ids` field partially does (a decision linked to zero assertions provably cannot move any probe's score). That structural fragment is the only part of our equivalence that behaves like a homomorphism — and it exists only where the QA artifact exists.

## Design question it bears on

Should the deployment-time equivalence mechanism be a *learned predictor* (v15's critic) or an *enforced constraint*? This paper's position: whatever fraction of the equivalence is derivable from declared structure should be enforced, never predicted, because enforcement generalizes to unseen instances by construction while prediction generalizes only as far as the evidence distribution covers. Our follow-up should therefore split the problem — derive the structurally provable part, and only ask the critic to cover the residue.

## Caveats

- Requires the symmetry group to be specified in advance; there is no discovery procedure here.
- Group-structured symmetry is a strong assumption. Our generalization lattice is a partial order with directional moves, not a group; the closest formal home would be an approximate/lax MDP homomorphism, which loses the exactness that makes this paper's guarantee attractive.
- The paper's gains are measured as sample efficiency on small control tasks, not as correctness of a tie-breaking rule under a secondary objective. Nobody here routes a second objective through the equivalence classes — the same gap flagged on [baram2021_action_redundancy.md](baram2021_action_redundancy.md).

## Sources

- [arXiv 2006.16908](https://arxiv.org/abs/2006.16908)
- [NeurIPS 2020 proceedings](https://papers.nips.cc/paper/2020/hash/2be5f9c2e3620eb73c2972d7552b6cb5-Abstract.html)
- Related in this wiki: [asadi2019_state_action_equivalence.md](asadi2019_state_action_equivalence.md) ([arXiv 1910.04077](https://arxiv.org/abs/1910.04077)) — the online-*estimated* equivalence counterpart; [kemertas2021_robust_bisimulation_metric.md](kemertas2021_robust_bisimulation_metric.md) ([arXiv 2110.14096](https://arxiv.org/abs/2110.14096)) — the learned-metric counterpart and its failure mode.
