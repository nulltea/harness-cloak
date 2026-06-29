---
type: research
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [inferdpt, rantext, differential-privacy, ldp, perturbation]
---

# RANTEXT — RANdom adjacency list TEXT perturbation

RANTEXT is the perturbation mechanism of InferDPT (Tong et al., 2023). It rewrites a
private document `Doc` into a perturbed document `Doc_p` by replacing **every**
in-vocabulary token with a semantically-near token sampled under ε-local differential
privacy. `Doc_p` is what gets sent to the remote black-box LLM; the raw `Doc` never
leaves the client.

Source: paper §V (Algorithm 2); reference `perturb_sentence` in
`github.com/mengtong0110/InferDPT` `func.py`.

## Definitions

| Term | Meaning |
|---|---|
| **`Doc` / `Doc_p`** | Raw document `⟨x₁…x_L⟩` / its perturbed version `⟨r₁…r_L⟩`. |
| **`V`** | Token vocabulary — the curated word-list RANTEXT perturbs over (paper: first ~11k English cl100k tokens). Tokens outside `V` are discarded. |
| **`φ(·)`** | Embedding function `V → ℝ^N`, mapping a token to a vector. |
| **`d_e(a,b)`** | Euclidean distance `√Σ(aᵢ−bᵢ)²`. |
| **`Δφ`** | Sensitivity of `φ` — the bounded coordinate range of the embedding space. Sets the Laplace noise scale; must be finite for the DP guarantee. |
| **`ε`** | Privacy budget (`ε ≥ 0`). Smaller ε = stronger privacy, weaker utility. |
| **ε-LDP** | ε-local differential privacy: a randomized `M` is ε-LDP if for any inputs `x,x'` and output `y`, `Pr[M(x)=y] / Pr[M(x')=y] ≤ e^ε`. Perturbation happens locally, before upload, with no trusted aggregator. |
| **Exponential mechanism** | DP primitive that samples output `y` with `Pr[y] ∝ exp(ε·u(x,y) / 2Δu)` for a scoring function `u`. |
| **`Ĉ(t)` / random embedding** | `φ̂(t) = φ(t) + Y`, the token embedding plus a Laplace noise vector `Y`. |
| **`C_e(t)`** | Random adjacent **embeddings**: all points within radius `d_e(φ̂(t), φ(t))` of `φ(t)`. |
| **`C_r(t)`** | Random adjacency **list**: the tokens whose embeddings fall in `C_e(t)` — the candidate replacements for `t`. |
| **`u(x,y)`** | Scoring function `1 − d_e(φ(x),φ(y)) / threshold` ∈ (0,1]; closer candidate → higher score. `Δu = 1`. |
| **`Z(ε)`** | Calibration factor for the noise scale (see §"ε is used twice"). |

## Overview

The innovation is the **random adjacency list**: instead of a fixed candidate set
(SANTEXT+ uses the whole vocab; CUSTEXT+ uses ~20 fixed neighbours), RANTEXT draws a
*random radius* per token from a Laplace distribution and takes every vocabulary token
inside that ball as the candidate set, then samples one via the exponential mechanism.

## Phase 0 — offline precompute (once per vocabulary)

1. Choose `V` and embed every token: `φ(t)` for all `t ∈ V`.
2. For each `t`, compute `d_e(φ(t), φ(t'))` to all other tokens and store them
   **sorted ascending**. This turns the online candidate lookup into one binary search.
3. Compute the embedding sensitivity `Δφ` (bounded coordinate range).

## Phase 1 — per-token perturbation (online)

Tokenise `Doc = ⟨x₁…x_L⟩`. For each token `xᵢ`:

1. **Filter / guards.**
   - `xᵢ ∉ V` → **discard** (no out-of-vocab / proper-noun leakage).
   - numeric → replace with a random number (reference: `randint(1,1000)`), then skip.
2. **Embed:** `eb_t = φ(xᵢ)`.
3. **Random radius.** Draw a Laplace noise vector `Y` (scale from `Δφ`, `ε`), add it:
   `eb_n = eb_t + Y`. Set `threshold = ‖eb_n − eb_t‖` — the radius is the *length of the
   noise*, so it is **random per token**.
   - small `ε` → large noise → large radius → many candidates (more privacy)
   - large `ε` → small noise → small radius → only close synonyms (more utility)
4. **Random adjacency list.** `C_r(xᵢ) = { t' ∈ V : d_e(φ(xᵢ), φ(t')) < threshold }`
   (a `searchsorted` cutoff on the precomputed sorted distances).
5. **Score each candidate:** `u(xᵢ, t') = 1 − d_e(φ(xᵢ), φ(t')) / threshold`.
6. **Exponential mechanism → probabilities:** `p(t') ∝ exp((ε/2)·u(xᵢ, t'))`, normalised
   over `C_r(xᵢ)`.
7. **Sample** the replacement `rᵢ ∼ p(·)` and append it.

Concatenate `⟨r₁…r_L⟩` → `Doc_p`.

## ε is used twice (and why `Z(ε)` exists)

`ε` controls both (a) the **radius** via the Laplace noise and (b) the **sampling
sharpness** `exp(ε/2·u)`. Because the radius is itself random, naive accounting would
break the guarantee, so the noise scale uses a calibrated factor:

```
Z = ε                       if ε < 2
Z = a·log(b·ε + c) + d       otherwise   (a≈0.0165, b≈19.0648, c≈−38.1294, d≈9.3111)
```

This calibration is what lets the paper prove **Theorem 2: per-token sampling is ε-LDP**.
Each token is perturbed **exactly once**, so there is **no composition** — the budget per
token stays `ε` across the whole document.

## Comparison to baselines

- **vs SANTEXT+** (static list = whole vocab): RANTEXT's radius is usually smaller →
  replacements stay relevant → better utility. SANTEXT+ also leaves a proportion of
  tokens unperturbed (a leak).
- **vs CUSTEXT+** (static list ≈ 20 fixed neighbours): RANTEXT's list is larger and
  randomised; Theorem 1 guarantees *any* `t' ∈ V` is reachable with some probability →
  much harder to invert. CUSTEXT+'s small list makes raw→raw self-mapping likely.

## Limitation: no sensitivity detector

**RANTEXT has no notion of which words are sensitive. It perturbs every in-vocabulary
token unconditionally** — there is no classifier deciding "private vs. safe."

This is deliberate, and the paper's reasoning is:

- Sensitivity is undecidable in general (a name, date, disease, relationship, or location
  can all be private depending on context), and any token left untouched is a free anchor
  for the adversary. So the DP stance is: protect everything uniformly rather than guess.
- It explicitly criticises detection-based methods (e.g. HaS) for "only protecting
  specific words of private entities, leaving others (not detected) exposed."
- It criticises SANTEXT+/CUSTEXT+ for leaving tokens effectively unperturbed
  (partial perturbation = leakage).

**Consequences:**

- *Benefit:* a clean, uniform per-token ε-LDP guarantee with no detector blind spots and
  no anchors; no composition (each token perturbed once).
- *Cost:* utility — non-sensitive words are scrambled too. InferDPT's extraction module
  exists precisely to repair this by regenerating fluent, aligned text from `Gen_p` plus
  the true prefix.
- *Cost:* out-of-vocab proper nouns are dropped entirely (privacy by removal), which
  removes key information and is one reason RANTEXT trails CUSTEXT+ on MAUVE on
  proper-noun-heavy datasets.

**Improvement direction (deviation from the paper):** *selective* perturbation — detect
and perturb only sensitive spans, leaving benign tokens intact for higher utility — is a
candidate iteration on top of InferDPT, with the obvious risk of detector blind spots
re-introducing leakage.
