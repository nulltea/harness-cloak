# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

_Field gaps with stable IDs._

## G1 — Curse of dimensionality in metric/distance-based DP-text mechanisms

**Status:** unresolved (active in this project)

Additive-noise and distance-radius token-level DP mechanisms (d_X-privacy / metric-LDP)
degrade as embedding dimensionality grows: pairwise distances and the noise norm both
**concentrate**, so a fixed-radius neighbourhood becomes all-or-nothing and the
exponential-mechanism scores (which use distance *ratios*) flatten toward uniform →
near-random replacement. Documented as the "curse of dimensionality" for d_X-privacy
text; the RANTEXT/InferDPT paper relies on a well-spread (ada-002) geometry without
naming it as a precondition. Observed first-hand here: qwen3-embedding single-token
vectors (1024-d, unit-norm) gave `|C_r| = 100%` of V at every ε.

Linked papers: paper:asghar2024_dxprivacy_text_curse, paper:feyisetan2019_privacy_utilitypreserving_textual,
paper:xu2020_differentially_private_text, paper:tong2023_inferdpt_privacypreserving_inference

## G2 — Density variation / metric conditioning of the embedding space

**Status:** unresolved (active in this project)

Word density varies across the embedding space and
## Failed Ideas (avoid repeating)
- **Use an LLM input-embedding matrix as φ to fix RANTEXT geometry**: 
- **Naive RANTEXT with qwen3-embedding as φ**: 
## Key Papers (102 total)
- [paper:abdolmaleki2020_distributional_view_multiobjective] A Distributional View on Multi-Objective Policy Optimization
- [paper:achiam2017_constrained_policy_optimization] Constrained Policy Optimization
- [paper:ambadkar2026_d3po_diversity_regularizer] D3PO: Decomposed, Diversity-Driven Policy Optimization for Preference-Conditioned MORL
- [paper:andrychowicz2020_what_matters_onpolicy_rl] What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
- [paper:angelopoulos2024_conformal_risk_control] Conformal Risk Control
- [paper:asadi2019_state_action_equivalence] Model-Based Reinforcement Learning Exploiting State-Action Equivalence
- [paper:asghar2024_dxprivacy_text_curse] $d_X$-Privacy for Text and the Curse of Dimensionality
- [paper:baram2021_action_redundancy] Action Redundancy in Reinforcement Learning
- [paper:barber2023_conformal_beyond_exchangeability] Conformal Prediction Beyond Exchangeability
- [paper:bartok2014_partial_monitoring_classification] Partial Monitoring—Classification, Regret Bounds, and Algorithms
- [paper:bates2021_risk_controlling_prediction_sets] Distribution-Free, Risk-Controlling Prediction Sets
- [paper:benabacha2023_mtsdialog_clinical_note] An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters (MTS-Dialog)
## Recent Relationships (54 total)
  paper:igamberdiev2023_dp_bart --extends--> paper:yue2021_differential_privacy_text
  paper:meisenbacher2024_dp_mlm --extends--> paper:igamberdiev2023_dp_bart
  paper:habernal2021_dp_nlp_devil --contradicts--> paper:krishna2021_adept
  paper:igamberdiev2023_dp_bart --extends--> paper:krishna2021_adept
  paper:meisenbacher2025_dp_st --extends--> paper:meisenbacher2024_dp_mlm
  paper:meisenbacher2024_1diffractor --extends--> paper:feyisetan2019_privacy_utilitypreserving_textual
  paper:meisenbacher2024_just_rewrite_again --extends--> paper:meisenbacher2024_dp_mlm
  paper:meisenbacher2025_spend_budget_wisely --extends--> paper:meisenbacher2024_dp_mlm
  paper:li2025_dp_gtr --extends--> paper:utpala2023_locally_differentially_private
  paper:yang2025_rupta --extends--> paper:staab2024_llm_anonymizers
  paper:igamberdiev2022_dp_rewrite --contradicts--> paper:krishna2021_adept
  paper:igamberd
