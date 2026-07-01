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

φ is **not an end-to-end utility lever** — final utility is flat (~0.89–0.93) across every φ,
because extraction re-grounds on the true prefix. Its **one established effect** is:

1. **A calibration constraint.** The noise radius (via `Z(ε)`) must be re-fit to each φ's
   distance scale, or the candidate set `|C_r|` degenerates and `Doc_p` becomes word-salad.
   The paper's `Z` constants fit only ada-002.

The coherence↔leakage "trade" seen *across* embeddings (ada-002 vs pythia, §3) is a **confound**
— vocabulary coverage, effective dimension, embedding family — **not anisotropy**. The controlled
test (same matrix raw vs mean-centered, matched `|C_r|`, §3a) does **not** reproduce it and the
leakage signs reverse. So **anisotropy is not a privacy↔coherence dial**; it only mattered because
it changes the distance scale that the radius calibration (effect 1) must be matched to.

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

## 3. A coherence↔leakage difference between the two φ — correlated with anisotropy, confounded

Both φ *targeted* `|C_r|=1%`, N=60, cased CNN/DM. Lower `inv@10`/`PII-leak` = more private;
higher `coherence` = more faithful `Doc_p`. Realized `|C_r|` is shown because it is **not**
actually matched per ε (see confounds).

| φ (aniso · |V| · eff_dim) | ε | realized \|C_r\| | coherence | inv@10 ↓ | PII-leak ↓ | utility (final) |
|---|---|---|---|---|---|---|
| ada-002 (0.79 · 10129 · 116) | 2 | 0.8% | 0.389 | 0.452 | 0.046 | 0.892 |
| ada-002 | 6 | 1.4% | 0.448 | 0.590 | 0.086 | 0.894 |
| ada-002 | 10 | 1.4% | 0.578 | 0.790 | 0.114 | 0.921 |
| pythia-410m (0.011 · 12000 · 570) | 2 | 1.2% | 0.535 | 0.666 | 0.129 | 0.899 |
| pythia-410m | 6 | 1.1% | 0.589 | 0.770 | 0.246 | 0.901 |
| pythia-410m | 10 | 0.7% | 0.716 | 0.912 | 0.478 | 0.929 |

**Observed (descriptive only).** The lower-anisotropy map (pythia) is more coherent yet more
invertible and leakier at every ε; `token_MI` (~4.6→5.0 bits) and final utility (~0.9) are tied
across both. Note **utility does not move** — φ is not a utility lever (§4); the difference is
coherence↔leakage, not utility↔leakage.

**The cause is NOT established (result-to-claim verdict: NO, high confidence).** This is a
2-point comparison of non-matched systems; at least four factors move *with* anisotropy and each
alone can produce the observed signs:

- **Vocabulary coverage.** ada's `|V|` is smaller (10129 vs 12000) and lacks cased name tokens,
  so it **drops** them (privacy-by-removal) — mechanically lowering ada's `PII-leak` and coherence
  with no geometry involved. Tell: the `PII-leak` gap *grows* with ε (0.08→0.43), the signature of
  a coverage/`|C_r|` artifact, not a fixed geometric offset.
- **`|C_r|` not actually matched.** Realized `|C_r|` (ada/pythia) ranges 0.67× to **2.0×**
  (ε=10: ada 1.4% vs pythia 0.7%). More candidates = more perturbation → directly moves both
  coherence and leakage.
- **Effective dimension** differs 5× (116 vs 570) — a separate concentration axis.
- **Bundled systems** — contrastive text-embedder vs LM `embed_tokens`, dim 1536 vs 1024,
  different tokenizer/coverage. Anisotropy is one of several simultaneous differences.

So `inv@10` is still suspect as a cross-φ privacy metric (it can be attack-weakening from
concentrated geometry), but attributing the trade to **anisotropy specifically** is a
*hypothesis* — **tested and refuted** in §3a.

**Plausible mechanism (hypothesis).** Removing the shared common direction raises distance/rank
*contrast* among neighbours, which would simultaneously (a) improve semantic neighbour selection
→ ↑coherence and (b) sharpen the neighbourhood constraint on the original token → ↑inversion.
This predicts the observed double-edge — but the 2-point data cannot separate it from the
confounds above, and the controlled test (§3a) **refutes** it.

## 3a. Isolating anisotropy — the controlled test → **mechanism REFUTED**

Held *everything but anisotropy* fixed: **qwen3-embedding RAW** (aniso 0.613) vs the **same
matrix mean-centered** (aniso 0.000) — identical vocabulary/token-set (12000), dim, tokenizer,
docs, seed, pipeline, attacker; centering only removes the shared direction. Realized `|C_r|`
matched to ~1.3% at ε=2/6/10 (ε=14 came out unmatched, 0.8% vs 1.4% — discounted). N=10, single
seed → directional, not tight. Δ = centered(iso) − raw(aniso), `results/dp_sweep_qwen_{raw,centered}.json`:

| ε | \|C_r\| raw/cen | coherence Δ | inv@10 Δ | pii_leak Δ | utility |
|---|---|---|---|---|---|
| 2 | 1.2% / 1.5% | −0.031 | −0.20 | −0.07 | ≈equal |
| 6 | 1.4% / 1.4% | −0.112 | −0.24 | −0.13 | ≈equal |
| 10 | 1.3% / 1.3% | +0.007 | −0.14 | −0.20 | ≈equal |
| *14* | *0.8% / 1.4%* | *+0.012* | *+0.07* | *−0.12* | *(|C_r| unmatched)* |

**Verdict: the §2-hypothesis is refuted.** Isolating anisotropy does **not** reproduce the
§3 (ada-vs-pythia) trade:

- **Coherence:** no consistent effect (mixed, ≈0 net). The pythia coherence edge in §3 was the
  vocabulary-coverage / `|C_r|` confound, not anisotropy. (Centering's *visible* salad-fix earlier
  in the project was escaping `|C_r|` saturation at the mis-set radius — a *calibration* effect,
  §2 — not an intrinsic coherence gain at matched `|C_r|`.)
- **inv@10 and PII-leak:** centering (more isotropic) moves them the **opposite** way from §3 —
  it *lowers* both. So the §3 signs ("isotropic → higher inv@10/PII-leak") were confound-driven
  (vocab coverage, eff_dim, embedding family), **not** anisotropy. This also refutes the older
  "anisotropy weakens the inversion attack" reading.
- **Utility:** unchanged — φ is not a utility lever (§4 holds).

So **anisotropy is not the cause of a coherence↔attackability trade.** At matched `|C_r|`,
removing it costs nothing on coherence/utility and *slightly reduces* leakage/attackability —
but that direction is weak (N=10, one seed; ε=14 `|C_r|` unmatched and flips inv@10). Firming it
needs N≥30, multiple seeds, CIs, and a dose-response (subtract α·mean); and it is within-qwen
only, not across embedding families.

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

- **Isolate anisotropy** — the raw-vs-centered qwen control (§3a) at matched realized `|C_r|`;
  this is the decisive experiment and is the current blocker on any causal claim.
- A context-aware LM reconstruction attack would test whether the anisotropy "protection"
  (low `inv@10`) survives a stronger adversary, or is only nearest-neighbour-attack weakening.
- Single seed, N=60; MAUVE needs N≥234 to be valid (the current runs are under-powered for it).

## Probes → code
`diagnostics.anisotropy` / `concentration` (aniso, dynamic range), `diagnostics.mechanism`
(`|C_r|`), `rantext.calibrate_noise_fn` (per-φ radius calibration), `attacks/
embedding_inversion.invert` (`inv@K`), `probes/leakage` (PII, S_w/N_w), `probes/mi`
(token-channel MI), `scripts/dp_sweep.py` (e2e sweep).
