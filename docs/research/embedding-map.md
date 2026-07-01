---
type: research
status: current
created: 2026-06-29
updated: 2026-07-01
tags: [lever2, phi, embeddings, rantext, inferdpt, anisotropy, noise-radius, privacy, utility]
companion: docs/research/rantext.md
---

> Rewritten 2026-07-01 from the faithful N=60 per-φ-calibrated runs. The prior 5-doc,
> single-seed A/B (the `noise_scale≈0.37` fudge, `syn_prec`/retention tables, jina/glove/
> phrasebert geometry sweep) was built on a miscalibrated radius and is discarded.

# Lever 2 — the embedding map φ

RANTEXT perturbs a token by sampling from a random-radius Euclidean ball in an embedding
space **φ: token → ℝ^N** (see `rantext.md`). φ does **not** enter the ε-LDP guarantee — that
is the exponential mechanism alone. φ enters the mechanism **only through Euclidean distances
between token vectors**, so its entire influence is carried by one property: the **distance
scale**. This doc records what that buys, from the faithful N=60 runs (per-φ calibrated radius,
cased CNN/DM, cl100k vocabulary, `results/dp_sweep_{ada002,pythia410m}.json`).

## Bottom line

φ is **not an end-to-end utility lever** — final extracted utility is flat (~0.89–0.93) across
every φ, because the extraction module re-grounds on the true prefix. φ has exactly two effects:

1. **A calibration constraint.** The noise radius (via `Z(ε)`) must be re-fit to each φ's
   distance scale, or the candidate set `|C_r|` degenerates and `Doc_p` becomes word-salad.
2. **A coherence↔attack-resistance trade, governed by anisotropy.** At a *matched* `|C_r|`,
   an isotropic φ gives more coherent `Doc_p` but a working inversion attack; an anisotropic
   φ gives less coherent `Doc_p` but the nearest-neighbour attack degrades — a lower `inv@10`
   that is an **artifact of compressed geometry, not real privacy**.

## Definitions

| Term | Meaning |
|---|---|
| **φ(·)** | The embedding map RANTEXT perturbs over. *This doc's subject.* |
| **distance scale / dynamic range** | The spread of pairwise Euclidean distances in φ. RANTEXT needs a usable range so a random radius admits a *graded* candidate set. |
| **anisotropy (`aniso_cos`)** | Mean cosine of random token pairs. 0 = isotropic; large = all vectors share a common direction, which compresses distances into a thin shell (small dynamic range). |
| **`|C_r|/V`** | Candidate-set fraction: share of the vocabulary inside the random radius. The operating point; set by the noise radius, held equal across φ for a fair A/B. |
| **`inv@10`** | Embedding-inversion attack: raw token among the 10 nearest to the perturbed token in φ (attacker knows φ). Privacy = 1 − inv. |
| **`coherence (Gen_p)`** | Coherence of the remote model's generation on `Doc_p` — the perturbation-side utility proxy. |
| **`PII-leak`** | Max cosine of each raw PII span to `Doc_p` (Presidio spans); raw sensitive semantics surviving into the perturbed prompt. |
| **utility (final)** | cos(Doc, extracted output) — the deliverable metric, downstream of extraction. |

## 1. φ enters only through distances → the distance scale is the whole story

Anisotropy is the property that sets the scale. Measured `aniso_cos`: **ada-002 0.79**,
**qwen3-embedding 0.61**, **pythia-410m 0.011**. A large `aniso_cos` means every vector shares
a big common component, so pairwise distances collapse into a narrow shell (nearest-neighbour
~0.5, random-pair ~0.65 for ada-002) — little dynamic range. An isotropic φ (pythia) spreads
distances out. Everything below follows from this one number.

## 2. The noise radius must be calibrated per φ (else the mechanism degenerates)

`Z(ε)`, which sets the radius, is a `scipy.curve_fit` to **one token in ada-002** (paper
Appendix B; see `rantext.md`). Those constants encode ada-002's shell and **do not transfer**:
on an anisotropic φ the fixed radius swallows a large `|C_r|`, sampling goes near-uniform, and
`Doc_p` is word-salad. This even breaks ada-002 *itself* (|C_r| 13–39% with its own constants
→ salad; the public repo does not reproduce its Table II).

The fix is `rantext.calibrate_noise_fn`: re-fit `Z` per φ to a fixed `|C_r|` target. The radius
— not the embedding — decides coherence. On pythia-410m, one prompt at ε=6 as `|C_r|/V` shrinks:

| `|C_r|/V` | `Doc_p` |
|---|---|
| 57% | `polymer Development templates his Portuguese University samsung women…` (salad) |
| 19% | `MY style vase she Bennett car amigos character moderate…` (mostly salad) |
| 4%  | `Mi name avis She Phillips and amigos several thirty became old living within Japan…` |
| 1%  | `Her name ais William Thompson … living by Canada with mead husband` (readable) |

## 3. At a matched `|C_r|`, anisotropy trades coherence for attack-resistance

Both φ calibrated to `|C_r|≈1%`, N=60, cased CNN/DM. Lower `inv@10`/`PII-leak` = more private;
higher `coherence`/`utility` = more faithful.

| φ (aniso) | ε | coherence (Gen_p) | inv@10 ↓ | PII-leak ↓ | utility (final) |
|---|---|---|---|---|---|
| ada-002 (0.79) | 2 | 0.389 | **0.452** | **0.046** | 0.892 |
| ada-002 (0.79) | 6 | 0.448 | **0.590** | **0.086** | 0.894 |
| ada-002 (0.79) | 10 | 0.578 | **0.790** | **0.114** | 0.921 |
| pythia-410m (0.011) | 2 | 0.535 | 0.666 | 0.129 | 0.899 |
| pythia-410m (0.011) | 6 | 0.589 | 0.770 | 0.246 | 0.901 |
| pythia-410m (0.011) | 10 | 0.716 | 0.912 | 0.478 | 0.929 |

**Reading.** At every ε, the isotropic map (pythia) is **more coherent** yet **more invertible
and leakier**; the anisotropic map (ada-002) is **less coherent** yet shows **lower `inv@10`
and `PII-leak`**. Crucially the token-channel MI is ~identical across φ (≈4.64→5.02 bits both):
the *channel capacity* is the same — the difference is only whether the nearest-neighbour attack
can exploit it. So a low `inv@10` on an anisotropic φ is **attack-weakening from compressed
distances, not genuine privacy**. `inv@10` is therefore not a clean cross-φ privacy metric.

## 4. End-to-end utility is φ-independent

Final utility (`cos(Doc, output)`, reranker utility, PII-reconstruction recall) is flat across
φ and near-flat across ε (utility 0.89–0.93; reranker/PII-recon ~1.0). The trusted extraction
model re-grounds on the true `Doc`, so it reproduces the content regardless of how `Doc_p`
looks. **φ cannot buy end-to-end utility.**

## 5. Neighbourhood quality and vocabulary coverage

- **Neighbourhoods are sensible in strong φ.** Nearest-neighbour sanity: `Boston→Baltimore/
  Chicago/Philadelphia`, `insulin→glucose/diabetes`, `Sarah→Sara/Laura/Emma`, `doctor→
  physician` — for ada-002, pythia-410m, and qwen3-embedding alike. So bad replacements come
  from the radius/anisotropy, not from bad neighbourhoods.
- **LM input-embedding matrices can be orthographically contaminated.** Qwen3-4B `embed_tokens`
  gives `insulin→Ins/INS/insulation/inscription` (keys partly on spelling); pythia-410m and the
  text embedders do not. Prefer a text/contrastive embedder or a clean LM matrix (pythia).
- **Vocabulary coverage decides drop-vs-substitute for names.** ada-002's shipped file is
  **10,129** tokens; our caches are **12,000** cl100k tokens. Cased single-token names
  (`Sarah`, `Johnson`, `Chicago`) are in the 12k but *not* ada's 10k → ada **deletes** names
  (privacy-by-removal), our φ **substitute** name→name (`Sarah Johnson→George Wilson`). This is
  a property of V, not φ geometry.

## Open

- `inv@10` confounds privacy with anisotropy; a matched-anisotropy comparison or a
  context-aware LM reconstruction attack is needed to test whether the anisotropy "protection"
  is real or evaporates against a stronger adversary.
- Single seed, N=60; MAUVE needs N≥234 to be valid (the current runs are under-powered for it).
- The coherence↔`inv@10` trade is measured only for ada-002 (anisotropic) vs pythia-410m
  (isotropic); a mid-anisotropy φ would fill in the curve.

## Probes → code
`diagnostics.anisotropy` / `concentration` (aniso, dynamic range), `diagnostics.mechanism`
(`|C_r|`), `rantext.calibrate_noise_fn` (per-φ radius calibration), `attacks/
embedding_inversion.invert` (`inv@K`), `probes/leakage` (PII, S_w/N_w), `probes/mi`
(token-channel MI), `scripts/dp_sweep.py` (e2e sweep).
