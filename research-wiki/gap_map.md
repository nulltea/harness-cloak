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

Word density varies across the embedding space and LM/sentence embeddings are
**anisotropic** (narrow-cone / representation degeneration), so isotropic Euclidean
distance is a poorly-conditioned basis for "semantic neighbourhood." Sparse-region words
rarely get replaced; dense-region words get over-replaced. Documented fixes: elliptical
**Mahalanobis** noise (accounts for covariance), **whitening / all-but-the-top**
(restores isotropy), and **counter-fitted** vectors (synonym-structured geometry, why
CusText+ beat GloVe). RANTEXT's `u = 1 − d/threshold` inherits this conditioning problem.

Linked papers: paper:xu2020_differentially_private_text, paper:chen2022_customized_text_sanitization,
paper:tong2023_inferdpt_privacypreserving_inference

## G3 — Privacy-preserving inference under a strict black-box (API-only, text-in/out) constraint

**Status:** partially addressed

Methods that need model internals (logits, embedding tables, two generations) cannot run
against a closed commercial API that returns only text. RANTEXT/InferDPT fit the
constraint (text perturb → remote → local reconstruct); newer SOTA analogs split on this:

- **Fit the scheme** (text-only, black-box): paper:tong2023_inferdpt_privacypreserving_inference (RANTEXT),
  paper:chen2023_hide_seek_has (entity hide/seek), paper:utpala2023_locally_differentially_private (DP-Prompt, DP paraphrase),
  plus token-DP mechanism swaps paper:yue2021_differential_privacy_text (SanText), paper:chen2022_customized_text_sanitization (CusText).
- **Break the scheme** (need white-box logits/embeddings): paper:thareja2025_dpfusion_tokenlevel_differentially (DP-Fusion, needs logits),
  paper:mai2023_splitanddenoise_protect_large (Split-and-Denoise, sends embeddings).
