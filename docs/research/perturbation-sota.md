---
type: research
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [inferdpt, rantext, perturbation, differential-privacy, embeddings, sota]
companion: docs/research/rantext.md
---

# RANTEXT perturbation: geometry limitations & SOTA alternatives

Terse map of why RANTEXT's perturbation degrades on modern embeddings, and the newer
mechanisms/frameworks that address it. Paper links point to the local research-wiki.
Gaps: [G1/G2/G3](../research-wiki/gap_map.md).

## The geometry RANTEXT assumes (and where it breaks)

RANTEXT replaces each token by sampling from a *random-radius Euclidean ball* in
embedding space, weighted by `u = 1 − d/threshold` (exponential mechanism). This silently
assumes the embedding space has **usable spread**: distances must separate "synonym" from
"random", and the radius must isolate a small neighbourhood. Three coupled failures:

| Issue | What goes wrong | Evidence |
|---|---|---|
| **Embedding geometric spread** (informal umbrella) | If all pairwise distances are nearly equal, the radius can't isolate a neighbourhood and `u` is ~constant → exponential mechanism collapses to **uniform sampling** → word-salad `Doc_p`. | Not named in [InferDPT](../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md); the paper implicitly relies on ada-002's spread (its Table V distances move with ε). |
| **Curse of dimensionality** (G1) | In high-d, both inter-token distances and the noise norm `‖Y‖` **concentrate** (rel. spread ~1/√d). Radius becomes all-or-nothing; ε barely modulates it. | Documented for d_X-privacy text ([curse paper](../research-wiki/papers/asghar2024_dxprivacy_text_curse.md), [Feyisetan](../research-wiki/papers/feyisetan2019_privacy_utilitypreserving_textual.md)). Our qwen3-embedding (1024-d, unit-norm) → `\|C_r\|=100%` of V at every ε. |
| **Density variation / metric conditioning** (G2) | Word density is uneven and embeddings are **anisotropic** (narrow cone). Isotropic Euclidean distance over-replaces dense-region words, under-replaces sparse ones. | [Mahalanobis paper](../research-wiki/papers/xu2020_differentially_private_text.md): spherical noise ignores covariance; fix is elliptical noise. |

Root cause for our build: **sentence-embedding models on single tokens are out-of-distribution**, compounding high-d concentration. RANTEXT inherits this because `u` uses distance *ratios*.

## SOTA alternatives — mechanism level (drop-in for the perturbation module)

All fit the black-box-API scheme (text-in/out).

| Mechanism | Fixes | Why better | Tradeoff |
|---|---|---|---|
| **Counter-fitted vectors + cosine** (used by [CusText](../research-wiki/papers/chen2022_customized_text_sanitization.md)) | spread, G2 | Synonym-injected/antonym-repelled static vectors → NN are *substitutable* words; cosine + exp-mechanism sidesteps Euclidean concentration | Fixed ~65k vocab (no OOV); English-only; word- not subword-level |
| **Mahalanobis / elliptical noise** ([Xu 2020](../research-wiki/papers/xu2020_differentially_private_text.md)) | G2, G1 | Noise shaped by embedding covariance → sparse words get replaced, dense words not over-replaced | Needs covariance estimate; still additive-noise (partial G1 relief) |
| **Whitening / all-but-the-top** (preprocessing) | spread, G2 | Removes the common-mode anisotropy cone → restores isotropy of any LM/sentence embedding in ~5 lines | Can over-flatten; tune # components |
| **Rank-based / truncated candidate set** (TEM-style, exp-mechanism) | G1, spread | Let noise pick a *list size k* (k-NN), not a metric ball → never empty/all; ignores absolute distance, immune to concentration | Departs slightly from RANTEXT's "random radius" derivation; re-prove ε-LDP |
| **SanText/SanText+** ([Yue 2021](../research-wiki/papers/yue2021_differential_privacy_text.md)) | — (baseline) | metric-LDP token swap | static whole-vocab list → low utility; needs a true metric |

## SOTA alternatives — framework level (the InferDPT role)

| Framework | Fits black-box API? | Why notable | Tradeoff |
|---|---|---|---|
| **HaS (Hide-and-Seek)** ([Chen 2023](../research-wiki/papers/chen2023_hide_seek_has.md)) | ✅ | Same protect→remote→restore shape; local model anonymizes *entities* then de-anonymizes | Detector blind spots = leakage; not DP |
| **DP-Prompt** ([Utpala 2023](../research-wiki/papers/utpala2023_locally_differentially_private.md)) | ✅ | Replaces token swap with **DP zero-shot paraphrase** → fluent `Doc_p`, no concentration issue | Sentence-level DP; needs a capable local paraphraser; weaker per-token guarantee |
| **DP-Fusion** ([Thareja 2025](../research-wiki/papers/thareja2025_dpfusion_tokenlevel_differentially.md)) | ❌ | Token-level DP over the *output* distribution (mix original vs redacted) → strong formal guarantee | Needs **logits + two generations** (white-box); not API-only |
| **Split-and-Denoise** ([Mai 2023](../research-wiki/papers/mai2023_splitanddenoise_protect_large.md)) | ❌ | Local denoise of LDP'd embeddings (extraction-module analog) | Sends **embeddings**, needs model internals |

## Recommendation for this project

1. **Counter-fitted + cosine** (mechanism swap) — biggest quality jump, no GPU, keeps RANTEXT's exp-mechanism. Fixes spread/G2 directly.
2. **Rank-based candidate set** — robustifies against G1 if any embedding still concentrates.
3. Keep **whitening** as a cheap toggle for the LM-matrix path.
4. Add **HaS / DP-Prompt** as alternative perturbation backends (baselines); **DP-Fusion / SnD** are out of scope (break black-box).
