---
type: research
status: current
created: 2026-06-29
updated: 2026-07-01
tags: [inferdpt, rantext, differential-privacy, ldp, perturbation, noise-radius]
companion: docs/research/embedding-map.md
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
| **`V`** | Token vocabulary — the **cl100k BPE tokens** RANTEXT perturbs over. Paper: first ~11k. Impl (`tokeniser.py`): the *same* cl100k tokenizer (tiktoken) both splits `Doc` (`surfaces`) and defines `V` (`english_vocab`, first ~12k tokens containing a letter), so surface tokens are looked up directly. One leading space is stripped from every token (matching the reference). Tokens outside `V` are discarded. |
| **`φ(·)`** | Embedding function `V → ℝ^N`, mapping a token to a vector. |
| **`d_e(a,b)`** | Euclidean distance `√Σ(aᵢ−bᵢ)²`. |
| **`Δφ`** | Sensitivity of `φ` — the **per-dimension** coordinate range vector `δ_d = max_t φ(t)_d − min_t φ(t)_d` over `V`. Each dim `d` calibrates its own Laplace scale `δ_d/Z(ε)`. **Not** a scalar `max_d δ_d`: collapsing to the max inflates the random radius by `δ_max/RMS(δ) ≈ 2.5–3×`, pushing it past the bounded unit-norm distance cloud → `|C_r|`→100% → uniform word salad. (Reference: `delta_f_new` per-dim vector in `func.py`.) |
| **`ε`** | Privacy budget (`ε ≥ 0`). Smaller ε = stronger privacy, weaker utility. |
| **ε-LDP** | ε-local differential privacy: a randomized `M` is ε-LDP if for any inputs `x,x'` and output `y`, `Pr[M(x)=y] / Pr[M(x')=y] ≤ e^ε`. Perturbation happens locally, before upload, with no trusted aggregator. |
| **Exponential mechanism** | DP primitive that samples output `y` with `Pr[y] ∝ exp(ε·u(x,y) / 2Δu)` for a scoring function `u`. |
| **`Ĉ(t)` / random embedding** | `φ̂(t) = φ(t) + Y`, the token embedding plus a Laplace noise vector `Y`. |
| **`C_e(t)`** | Random adjacent **embeddings**: all points within radius `d_e(φ̂(t), φ(t))` of `φ(t)`. |
| **`C_r(t)`** | Random adjacency **list**: the tokens whose embeddings fall in `C_e(t)` — the candidate replacements for `t`. |
| **`u(x,y)`** | Scoring function `1 − d_e(φ(x),φ(y)) / threshold` ∈ (0,1]; closer candidate → higher score. `Δu = 1`. |
| **`Z(ε)`** | Factor setting the Laplace noise scale `δ_d/Z` (hence the radius). A *utility* knob, not a privacy quantity — see §"`Z(ε)` is a utility knob" and §"Noise-radius calibration". |

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

Tokenise `Doc = ⟨x₁…x_L⟩` with the cl100k tokenizer (leading space stripped). Because it is
BPE and **cased**, a common cased word/name is usually one token (`" Boston"→"Boston"`) and is
perturbed whole; a lowercased or rare word fragments (`" boston"→["b","oston"]`), and each
fragment is perturbed or dropped independently. For each token `xᵢ`:

1. **Filter / guards.**
   - `xᵢ ∉ V` → **discard** (out-of-vocab / rarer proper-noun fragments removed).
   - numeric (`isdigit`) → replace with a random number (`randint(1,1000)`), then skip.
     *Only digit tokens; spelled-out numbers ("thirty four") are perturbed as ordinary words.*
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

## `Z(ε)` is a utility knob, NOT a privacy calibration

A common misreading (which this doc previously stated) is that `Z(ε)` is what makes the
mechanism ε-LDP. **It is not.** The proof of **Theorem 2** (paper Appendix C, Eq 27–32) is
the *standard exponential-mechanism argument*: for `x,x',y ∈ C_r(t)`,
`Pr[y|x]/Pr[y|x'] = [exp(ε·u(x,y)/2)/exp(ε·u(x',y)/2)]·[Σexp(ε·u(x',y')/2)/Σexp(ε·u(x,y')/2)]`,
and since `Δu=1` and `0<u≤1` each factor is `≤ exp(ε/2)`, so the product `≤ eᵋ`. **`Z`, the
radius, and the Laplace noise appear nowhere in the proof.** The ε-LDP guarantee comes
entirely from the exponential mechanism's `ε/2` factor; the radius/`C_r` is treated as fixed
and given. (The `/2` is the textbook EM factor — numerator and denominator both shift — *not*
"ε spent twice on radius + sampling.")

So `Z` does **not** affect the formal privacy at all. What it controls is the **radius →
`|C_r|` operating point** (bigger radius = more candidates = more empirical noise, less
utility). `Z` is defined in paper **Appendix B**: a `scipy.curve_fit` chosen so the ratio
`|C_r|/|V|` of the reference token **"happy"** in the paper's **ada-002** space hits a target
schedule (Table XI). The published constants

```
Z = ε                       if ε < 2
Z = a·log(b·ε + c) + d       otherwise   (a≈0.0165, b≈19.0648, c≈−38.1294, d≈9.3111)
```

therefore **encode ada-002's geometry around one token** and are *meaningless on any other
embedding*. Verified empirically: running the faithful mechanism on the paper's *own* ada-002
embeddings with these constants gives `|C_r|` 13–39% (token-dependent) → word-salad `Doc_p`,
i.e. the public repo does **not** reproduce its Table II.

Each token is perturbed **exactly once**, so there is **no composition** — the per-token
budget stays `ε` regardless of `Z`.

## Noise-radius calibration — the curve-fit algorithm

Because `Z` is a free utility knob, the radius must be *chosen*. Both the paper and our
implementation choose it by targeting a candidate-set fraction `|C_r|/|V|`.

**Paper (Appendix B).** Quote: *"generating adjacency lists directly with Laplace distribution
led to excessively large sizes. To tackle this, we created an adjusted random vector by
`curve_fit`, aiming to achieve specific probability targets for the ratio |C_r|/|V| of the
token 'happy'."* The procedure:

1. Pick one **reference token** (paper: `"happy"`).
2. Pick a target `|C_r|/|V|` schedule over ε (Table XI: ε = 2/6/10/14 → 1.5%/9.0%/10.0%/10.5%).
3. `scipy.curve_fit` the closed form `Z(ε) = a·log(b·ε + c) + d` so the reference token's
   `|C_r|` hits the schedule → the published `a,b,c,d`.

⇒ those constants encode **ada-002's** geometry around **one token**; they are not universal.

**Ours (`rantext.calibrate_noise_fn`, per-φ).** Same idea, re-fit to each φ, and exact rather
than a stochastic curve-fit. Key identity: `radius = ‖Laplace(0, Δφ)‖ / Z`, so the radius is a
deterministic function of `Z` once base draws are fixed:

1. Draw a fixed batch of `‖Laplace(0, Δφ)‖` samples (base radii); `radius(Z) = base / Z`.
2. For a small set of reference tokens, `|C_r|(Z)` is monotone-decreasing in `Z` → **bisect**
   `Z` so the mean `|C_r|/|V|` equals the target (default **1%**).
3. Return a constant `noise_fn(ε) = Z`. (The paper's `Z(ε)` is near-flat for ε≥2, so one `Z`
   reproduces it; `ε` then drives only the exponential-mechanism sharpness `exp(ε/2·u)`.)

For a cross-φ comparison, hold the **`|C_r|` target** equal across φ (equal operating point) —
never hold `Z` equal, since a fixed `Z` lands at different `|C_r|` per embedding.
See `embedding-map.md` (the empirical-honesty rule: fix ε + the |C_r| target, never a per-model fudge).

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
