# Initial Experiment Results — Lever 2 φ

**Date**: 2026-06-29
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Scope run**: M0 (build + probes), M2 Block 1 geometry/mechanism, B2 isolation. Vocab = whole-word
sub-vocab (7,913 tokens = 12k cl100k ∩ counter-fitted vocab, 66% coverage). CPU-only, no GPU.

## Results by Milestone

### M0: Sanity — PASSED
Built 4 row-aligned φ caches on the same 7,913-word sub-vocab: `vocab_qwen_sub` (1024-d, served),
`vocab_glove` (300-d), `vocab_cf` counter-fitted (300-d), `vocab_phrasebert` (768-d, CPU).
New probes added + run clean: `eff_dim` (participation ratio), `synonym_precision` (WordNet),
anisotropy-corrected retention.

### M2 Block 1 + B2: φ geometry/mechanism A/B at matched candidate load |C_r|≈0.05, ε=3
`results/phi_geometry_ab.json` — ret_corr = mean ± bootstrap CI over 8 seeds; syn_prec on a fixed
shared query set (7,486 words), paired bootstrap CI vs glove. (Round-1 review hardened this with CIs,
an equivalence margin, and the whitened-GloVe isotropy-only control.)

| φ | dim | rel_spread | eff_dim | aniso_cos | ret(raw) | **ret_corr [CI]** | **syn_prec@10 [CI]** | Δsyn vs glove [CI] |
|---|---|---|---|---|---|---|---|---|
| qwen3-emb (baseline) | 1024 | 0.086 | 137 | 0.633 | 0.81 | 0.453 [.442,.469] | 0.102 [.099,.104] | +0.003 [.001,.006] |
| glove (ref) | 300 | 0.074 | 84 | 0.162 | 0.53 | 0.446 [.437,.464] | 0.098 [.096,.101] | — |
| glove-whitened (isotropy-only) | 300 | 0.051 | 112 | ≈0 | — | **0.377 [.365,.392]** | 0.100 [.098,.103] | +0.002 [.001,.003] |
| **counter-fitted (P2-strong)** | 300 | 0.048 | 147 | 0.038 | 0.43 | 0.431 [.411,.451] | **0.159 [.156,.162]** | **+0.061 [.058,.063]** |
| Phrase-BERT (P4/neural) | 768 | 0.092 | 52 | 0.436 | 0.68 | 0.434 [.429,.442] | 0.111 [.108,.114] | +0.013 [.010,.015] |

Equivalence margin on ret_corr = 0.076 (NOT a tie within 0.03 — driven by the whitening outlier).

## Summary
- **2/2 must-run geometry blocks completed** (Block 1 geometry, B2 isolation). Main result: **mixed
  — one correction, one positive isolation, one new negative on isotropy.**
- **Finding 1 (correction, high-confidence):** *raw retention `cos(orig,repl)` is dominated by the
  anisotropy floor.* Corrected retention `(cos−floor)/(1−floor)` collapses the cross-**model** spread
  from 0.36 (raw) to ~0.02 (qwen/glove/cf/Phrase-BERT 0.43–0.45) — ~94% of the apparent gap was the
  floor. NOT a formal tie within 0.03 (margin 0.076): the **whitening transform** is a genuine outlier
  (0.377). So the matrix's prior "0.23 vs 0.79" gap was mostly the floor, not synonym quality.
- **Finding 1b (new, from the isotropy-only control):** whitened GloVe (aniso≈0) *lowers* corrected
  retention (0.446→0.377) and does **not** raise syn_prec (+0.002, CI [.001,.003]). **Isotropy alone is
  counterproductive, not merely insufficient** — removing the anisotropy cone degrades the mechanism.
- **Finding 2 (isolation, high-confidence):** on the floor-free P2 measure (WordNet syn_prec@10, paired
  over 7,486 shared queries), **only the counter-fitting retrofit helps** — cf +0.061 vs its own
  pre-retrofit glove (CI [.058,.063]) ≫ Phrase-BERT +0.013 > qwen +0.003 ≈ whitened-glove +0.002.
  Isotropy, low-eff-dim, and paraphrase-training do not reach P2; explicit synonym-fitting does.
- **Caveat:** absolute syn_prec stays low (cf ~1.6/10 WordNet neighbours). WordNet undercounts
  paraphrastic substitutes; an LLM-judge probe may widen the gap. φ-alone headroom looks bounded
  → tests whether e2e masks the difference (C3, re-running on GPU with seeds + per-doc CIs).

### M2 Block 1 e2e arm (C3) — `results/phi_e2e_ci.json` (host `.venv` GPU; 5 seeds × 5 docs = 25 paired)

Hardened per round-1 review: multi-seed paired design, **remote `Gen_p` (φ-sensitive) separated from
final extracted output**, paired bootstrap CIs, baseline = qwen. Scoped to the decisive contrast
(qwen vs counter-fitted — the only φ that moved P2). SimCSE scorer on the host `.venv` GPU.

| φ | remote `Gen_p` util | Δ vs qwen [CI] | final util | Δ vs qwen [CI] | pii_leak (n=15) | Δ vs qwen [CI] |
|---|---|---|---|---|---|---|
| qwen3-emb | 0.501 | 0 | 0.800 | 0 | 0.523 | 0 |
| counter-fitted | 0.482 | −0.019 [−0.066,+0.024] | 0.792 | −0.008 [−0.036,+0.019] | 0.559 | +0.036 [−0.011,+0.081] |

- **Finding 3 (C3, pilot evidence):** cf's +0.061 syn_prec edge **does not translate e2e** — Δ-vs-qwen
  CIs straddle 0 for the remote `Gen_p` (φ-sensitive stage), the final output, *and* PII leakage.
  Stronger than "the extractor masks it": φ is utility-irrelevant even at the remote stage (ε=3/|C_r|=5%
  perturbs heavily enough that output quality is φ-insensitive); the extractor then lifts both ~0.49→0.80,
  equalising them. Caveats: **pilot** — 5 docs × 5 seeds (=25 pairs; bootstrap over seed-doc pairs
  overstates independence with only 5 docs), so this is pilot evidence at ε=3, not a settled general
  e2e/leakage result. PII observed in 15/25 rows (an earlier `nan`-aggregation bug had implied 0); the
  cf−qwen PII delta is inconclusive. (Earlier 4-φ single-seed `e2e_ab_subvocab.json` agreed directionally.)

## Headline conclusion
**Lever 2 (φ) has low leverage on end-to-end utility** under InferDPT's perturb→remote→extract
architecture at ε=3 / |C_r|=5%. The geometry probes matter for *understanding* the mechanism (and
correcting the retention artifact), but swapping φ — even to a synonym-fit space — buys ≈0 e2e.
Effort should move to higher-leverage knobs: the ε/|C_r| operating point, the extraction module, or
selective perturbation.

## Review outcome (auto-review-loop, codex/gpt-5.5 xhigh)
- Round 1: 3/10 "Almost" — flagged retention "tie" without CIs, "matched privacy" overclaim,
  underpowered C3, missing isotropy-only control. → fixed: bootstrap CIs + equivalence margin,
  whitened-GloVe control, fixed-query paired syn_prec, multi-seed paired e2e (remote vs final), GPU.
- Round 2: **7/10 "Almost"** — stop condition met (≥6 ∧ almost). Termination cleanup applied:
  `nanmean` PII bug fix (15/25 rows had PII; my "no PII" claim was false → corrected), bootstrap
  ret-CI relabel, raw-retention + achieved-load stored, stale claim Statement superseded, wording
  narrowed ("materially", "this whitening control"). Full log: `review-stage/AUTO_REVIEW.md`.

Deferred (low expected value given C3): jina low-dim P1 control, fastText P6 coverage, rank-based
variant (B3), cluster-bootstrap e2e over more docs to firm up C3 beyond pilot.

## Next Step
→ `/auto-review-loop "Lever 2 φ: retention is an anisotropy artifact; φ choice is masked e2e"`

## result-to-claim verdict (Codex gpt-5.5 xhigh, 2026-06-30) — overall: PARTIAL
Provisional (no EXPERIMENT_AUDIT.json run; Codex judged on pre-verified numbers, could not self-read files under sandbox).

| Claim | Verdict | Revision |
|---|---|---|
| C1 raw retention = anisotropy artifact | partial | "strongly confounded; floor-correction removes most cross-model spread but is NOT a formal tie, and is degenerate for jina" |
| C2 only counter-fitting moves syn_prec | yes (caveats) | "only φ with a practically large gain; Phrase-BERT small-but-positive (+0.013); WordNet-favorable by construction" |
| C1b/P3 whitening isotropy-only counterproductive | yes (this control) | not a general isotropy claim — one whitening transform |
| P1 jina low-dim/dynamic-range insufficient | yes (negative control) | jina ret_corr≈1.0 is DEGENERATE (eff_dim 3.4) → corrected retention fails as a metric here |
| C3 cf edge doesn't reach e2e | partial | "no detected gain in pilot; underpowered (5 docs), not generalizable" |

**Inference gaps to close:** (1) cluster-bootstrap syn_prec by synset/POS (current CIs too narrow); (2) e2e over dozens of docs with doc-clustered bootstrap (lift C3 past pilot); (3) same-base controlled dimensionality sweep (PCA/random projection) to cleanly isolate P1; (4) non-WordNet / contextual-substitution synonym validation. None overturn the core story; they bound it.
