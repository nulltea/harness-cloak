---
type: research
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [mutual-information, leakage, rantext, v-information, context-leakage, probes]
companion: docs/perturbation-sota.md
---

# Mutual-information leakage probes for RANTEXT

Why MI (not `cos(Doc,Doc_p)`) is the right leakage measure, the per-token vs
context-aware distinction, estimation options, and what is implemented in
`src/inferdpt/probes/mi.py`.

## Definitions

| Term | Meaning |
|---|---|
| **X, Y** | raw token (sequence `X₁:ₙ`) and its perturbed counterpart `Y₁:ₙ` |
| **`p(y\|x)`** | the mechanism's channel. For RANTEXT, analytic: `E_noise[ softmax(ε/2·u) over C_r ]` |
| **`I(X;Y)`** | mutual information, `H(Y)−H(Y\|X)`; bits of `X` revealed by `Y`. 0 = independent |
| **`H(·)`** | Shannon entropy (bits here) |
| **V-information** `I_V(Y→X)` | usable info to a *bounded* predictor family `V`: `H_V(X)−H_V(X\|Y)` (Xu et al. 2020) |
| **DPI** | data-processing inequality; Shannon MI obeys it, V-information does not |
| **S_w / N_w** | self-substitution prob / output support — coarse functionals of `p(y\|x)` |
| **Memoryless channel** | `p(Y\|X)=Πᵢ p(Yᵢ\|Xᵢ)` — RANTEXT perturbs each token independently given `Xᵢ` |

## Why MI over `cos(Doc, Doc_p)`
`cos` is an arbitrary-scale heuristic (floored ~0.61 by qwen3 anisotropy) that conflates
utility- and privacy-relevant similarity. `I(X;Y)` is in **bits**, comparable across
mechanisms/ε, and via **Fano's inequality** *upper-bounds every adversary's*
reconstruction success — attacker-agnostic. Cuff & Yu (CCS 2016) formally tie DP to a
bounded-MI constraint, so MI is the natural *realized*-leakage companion to the ε *worst-case*
guarantee (realized MI is usually ≪ ε).

## (1) Per-token channel MI — `I(Xᵢ;Yᵢ)`  [implemented]
We **know** RANTEXT's channel, so we estimate `p(y|x)` by soft-averaging the analytic
exponential-mechanism conditional over noise draws (no hard sampling → low variance), then:

```
I(X;Y) = H(Y) − H(Y|X),   H(Y|X) = Σ_x p(x) H(p(y|x)),   p(y) = Σ_x p(x) p(y|x)
per-token leakage L(x) = KL(p(y|x) ‖ p(y))   # which tokens leak most → correlate with PII
```

`prior p(x)` is uniform (or unigram). This is **standard** for a known discrete channel
(textbook MI of a channel); CLUB/MINE are only needed when `p(y|x)` is intractable. It
generalizes `S_w/N_w` (`H(Y|X)` ≈ log effective output count; `S_w` is one channel entry).
**Limitation: per-token only — it misses cross-token context (below).**

## Why per-token MI misses context (precisely)
RANTEXT is **memoryless**, giving two facts:
- **Total** leakage is bounded by the per-token sum: `I(X₁:ₙ;Y₁:ₙ) ≤ Σᵢ I(Xᵢ;Yᵢ)`.
- **But recovery of a single token is under-counted:** `I(Xᵢ; Y₁:ₙ) ≥ I(Xᵢ; Yᵢ)`. Because the
  *source* `X` is correlated (language), an adversary infers `Xᵢ` from the **whole** output
  `Y₁:ₙ` + a language prior: `H(Xᵢ|Y₁:ₙ) ≪ H(Xᵢ|Yᵢ)`. That gap **is** context leakage, and
  it's what the LLM attack exploits.

## (2) Context-aware MI
### V-information (the honest framing)
Shannon MI assumes an unbounded adversary; a real one is a bounded predictor family `V`
(an LLM). **V-information** (Xu et al. 2020, [2002.10689](https://arxiv.org/abs/2002.10689))
`I_V(Y→X)=H_V(X)−H_V(X|Y)` **violates DPI — computation can create usable info** — which is
exactly why Qwen3.6-35B extracts more contextual leakage than BERT on the same fixed channel.

### Estimation menu
| Approach | Measures | Bound | Cost / caveat |
|---|---|---|---|
| **V-information via the LLM attacker** | usable contextual leakage to a realistic adversary | lower (tightens w/ attacker) | reuses LLM-recon; needs token logprobs or a recovery proxy |
| **MINE / InfoNCE** ([1801.04062](https://arxiv.org/abs/1801.04062)) | neural MI of `(X,Y)` embeddings | lower | train a critic; loose/high-variance |
| **CLUB** (Cheng 2020) | same | upper (conservative) | approximation gap |
| **LM conditional entropy / "context influence"** ([2410.03026](https://arxiv.org/abs/2410.03026)) | `H(Xᵢ\|Y₁:ₙ)` via an LM; with/without delta | either | DP-style with/without control |
| **Block / n-gram MI** ([2605.20187](https://arxiv.org/html/2605.20187)) | local windowed `I(X_blk;Y_blk)` | empirical | cheap, no LLM; needs many tokens for n≥2 |
| **Fano (attack) + CLUB** | brackets sequence MI | both | combine lower⊕upper |

### The critical confound
[Estimating Privacy Leakage of Augmented Contextual Knowledge (ACL'25)](https://arxiv.org/abs/2410.03026):
comparing an LLM's recovery directly to the source **overestimates** leakage — the LLM may
"recover" PII from its **parametric knowledge**, not from `Y`. Fix (DP-style "context
influence"): contextual leakage ≈ `H_V(X|control) − H_V(X|Doc_p)`, i.e. attacker NLL given a
**neutral prompt (no Doc_p)** minus given the perturbed sequence — the delta is what `Doc_p`
actually contributed. Always apply this with/without-`Y` control.

## (3) Block / n-gram MI — `I(X_blk;Y_blk)`  [implemented]
Empirical plug-in MI (Miller–Madow bias-corrected) between aligned raw and perturbed
n-grams over a corpus. `n=1` ≈ empirical token MI; `n≥2` captures *local* source+channel
dependence (a cheap, no-LLM context proxy). Needs many tokens for `n≥2` (sparsity).

## Recommendation for this project
1. **Intrinsic, now:** per-token channel MI (1) + n-gram MI (3) — both in `probes/mi.py`,
   no LLM, deterministic. Bracket the per-token MI as a near-upper intrinsic reference.
2. **Realistic, next:** V-information via the LLM-recon attacker with the no-context control
   → contextual leakage in bits (lower bound, tightens with attacker). Deferred until the
   LLM-recon attack lands; needs logprobs (else fall back to recovery-rate, a coarser bound).
3. Optionally CLUB on sentence embeddings for a variational upper bound (only worth it if we
   move to embedding-level perturbation).

## Caveats
- Lower bounds (attack/V-info) under-report and are attacker-dependent — always report *which* `V`.
- CLUB over-reports (approximation gap); exact sequence MI for text is intractable.
- Plug-in MI is biased upward at finite N (Miller–Madow mitigates); report `N`, support sizes.
- Per-token / n-gram MI are *mechanism-level intrinsic* measures — they do not capture a
  language-model adversary; that requires the V-information probe.
- MI is prior-dependent — report the input prior.

## What is implemented vs deferred
- **Implemented** (`probes/mi.py`): `token_channel_mi` (1), `ngram_mi` (3).
- **Deferred** (documented): V-information via LLM attacker + context-influence control;
  CLUB/MINE variational estimators.

## References
Xu et al. 2020 *A Theory of Usable Information* ([2002.10689](https://arxiv.org/abs/2002.10689)) ·
Cuff & Yu 2016 *DP as a Mutual Information Constraint* (CCS) ·
Belghazi et al. 2018 *MINE* ([1801.04062](https://arxiv.org/abs/1801.04062)) ·
Cheng et al. 2020 *CLUB* · van den Oord et al. 2018 *CPC/InfoNCE* ·
*Estimating Privacy Leakage of Augmented Contextual Knowledge* ([2410.03026](https://arxiv.org/abs/2410.03026)) ·
*Can LLMs Keep a Secret? / Contextual Integrity* ([2310.17884](https://arxiv.org/html/2310.17884v2)) ·
*Pairwise MI in Masked Discrete Sequence Models* ([2605.20187](https://arxiv.org/html/2605.20187)).
