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
## Key Papers (10 total)
- [paper:asghar2024_dxprivacy_text_curse] $d_X$-Privacy for Text and the Curse of Dimensionality
- [paper:chen2022_customized_text_sanitization] A Customized Text Sanitization Mechanism with Differential Privacy
- [paper:chen2023_hide_seek_has] Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection
- [paper:feyisetan2019_privacy_utilitypreserving_textual] Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations
- [paper:mai2023_splitanddenoise_protect_large] Split-and-Denoise: Protect large language model inference with local differential privacy
- [paper:thareja2025_dpfusion_tokenlevel_differentially] DP-Fusion: Token-Level Differentially Private Inference for Large Language Models
- [paper:tong2023_inferdpt_privacypreserving_inference] InferDPT: Privacy-Preserving Inference for Closed-box Large Language Model
- [paper:utpala2023_locally_differentially_private] Locally Differentially Private Document Generation Using Zero Shot Prompting
- [paper:xu2020_differentially_private_text] A Differentially Private Text Perturbation Method Using a Regularized Mahalanobis Metric
- [paper:yue2021_differential_privacy_text] Differential Privacy for Text Analytics via Natural Text Sanitization
## Recent Relationships (28 total)
  paper:xu2020_differentially_private_text --addresses_gap--> gap:G1
  paper:xu2020_differentially_private_text --addresses_gap--> gap:G2
  paper:chen2022_customized_text_sanitization --addresses_gap--> gap:G2
  paper:tong2023_inferdpt_privacypreserving_inference --addresses_gap--> gap:G3
  paper:chen2023_hide_seek_has --addresses_gap--> gap:G3
  paper:utpala2023_locally_differentially_private --addresses_gap--> gap:G3
  paper:yue2021_differential_privacy_text --addresses_gap--> gap:G3
  paper:thareja2025_dpfusion_tokenlevel_differentially --addresses_gap--> gap:G3
  paper:mai2023_splitanddenoise_protect_large --addresses_gap--> gap:G3
  idea:naive-rantext-qwen3 --inspired_by--> paper:tong2023_inferdpt_privacypreserving_inference
  idea:naive-rantext-qwen3 --addresses_gap--> gap:G1
  idea:naive-rantext-qwen3 --addresses_gap--> gap:G2
  idea:naive-rantext-qwen3 --tested_by--> exp:geom-dia
