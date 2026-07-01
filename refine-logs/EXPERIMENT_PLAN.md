# Experiment Plan — Lever 2: the embedding map φ for RANTEXT

**Problem**: RANTEXT's perturbation quality is bottlenecked by the geometry of φ (token→ℝ^N).
The current φ (qwen3-embedding-0.6B) survives only via a `noise_scale≈0.37` band-aid, and the
obvious isotropy fix (Qwen3-1.7B `embed_tokens` matrix) made retention *worse* (0.79→0.23).
**Method Thesis**: φ fitness is governed by **local synonym structure (P2)**, not isotropy (P3)
or anti-concentration (P1) — a φ purpose-built for synonym k-NN (counter-fitted) recovers
retention at matched privacy, where isotropy-only and low-dim-only do not.
**Date**: 2026-06-29
**Companion**: `docs/research/embedding-map.md` (P1–P7 + probe definitions), `claim:anisotropy-bad-but-insufficient`

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1 (primary)** P2 (local synonym structure) is the *binding* constraint on φ fitness | Redirects the whole φ search: stop chasing isotropy/low-dim, chase synonym geometry | At matched ε & matched `|C_r|/V` (eval.py `calibrate`) & checked leakage (S_w, token-MI, inv@K), **counter-fitted retention & k-NN-synonym-precision strictly > qwen3-emb and > every control**; φ ranking by retention tracks ranking by P2, NOT by isotropy/effective-dim | B1, B2 |
| **C2 (supporting)** the P2 gain is necessary-not-sufficient for its rivals | Rules out the anti-claims | Isotropy-only (matrix) and low-dim-only (jina-v3-32d) each fail to recover retention despite winning P3 / P1 respectively | B2 |
| **C3 (scope/honesty)** φ-retention gains may be *masked end-to-end* by the extraction LLM | Prior A/B showed e2e leakage tied & utility close — must report honestly, not oversell | e2e utility/leakage A/B; if φ gains wash out e2e, state it (and that the contribution is mechanism-level + attack-surface, not e2e utility) | B1, B5 |

**Anti-claims to rule out** (each gets a control φ):
- "gain is just isotropy" → **Qwen3-matrix** (isotropic, P2-poor) — already negative, re-confirm.
- "gain is just low dimensionality / anti-concentration" → **jina-v3 truncated 32–64d** (P1-best, P2-weak).
- "gain is just OOV coverage" → **fastText cc.en.300** (same static word class, full OOV, *no* synonym retrofit).
- "retention doesn't translate to the mechanism" → **rank-based candidate variant** (B3) + e2e (B1).

## Paper Storyline
- **Main paper must prove**: C1 (P2 binds) + C2 (P1/P3 each insufficient) — one main probe-panel table + one isolation figure.
- **Appendix can support**: B3 (rank-based mechanism variant), B4 (neural-vs-static), B5 (OOV + e2e-masking diagnosis).
- **Intentionally cut**: GloVe full sweep (dominated by fastText; keep only 50-d as an optional P1 ablation); jina-v2-small (same OOD class as qwen, no new signal); whitening/IsoScore as *candidates* (they're a post-hoc knob, not a model).

## Experiment Blocks

### Block 1: Main anchor — φ probe-panel A/B at matched privacy  ·  Priority: MUST-RUN
- **Claim tested**: C1, C3.
- **Why this block exists**: the one table that shows whether a synonym-structured φ beats the baseline on the metrics that decided every prior verdict.
- **Dataset / task**: vocab cache (12k cl100k English tokens, row-aligned across φ) for geometry/mechanism; `corpora/dev.txt` for e2e.
- **Compared systems (φ)**: qwen3-emb-0.6B (baseline) · counter-fitted (primary) · Phrase-BERT · fastText · jina-v3-32d · Qwen3-matrix (control). All on the **same word list** (intersection / skip-perturb OOV) for a fair A/B.
- **Metrics** (decisive first): **retention `cos(orig,repl)`**, **rel_spread (P1)**, **k-NN synonym precision (P2, new)**; then anisotropy-cos (P3), effective-dim, `|C_r|/V`, norm_entropy; leakage: S_w/N_w, `token_channel_mi`, inv@10; e2e: utility, utility_control, pii_recon, overlap.
- **Setup**: ε=3 (paper-faithful), `eval.py calibrate` to target `|C_r|/V=0.05` per backend (matches privacy), seed-averaged (3 seeds for mechanism/e2e). Reuse `eval.eval_backend`.
- **Success criterion**: counter-fitted retention > 0.788 AND k-NN-syn-precision highest, at matched `|C_r|` and non-worse S_w/MI/inv.
- **Failure interpretation**: if counter-fitted does *not* top retention, P2-binds hypothesis is wrong → revisit whether *any* single property explains fitness.
- **Table/figure target**: Table 1 (main).

### Block 2: Novelty isolation — which property explains the ranking?  ·  Priority: MUST-RUN
- **Claim tested**: C1, C2.
- **Why**: proves the *cause* is P2, not a confound. Counter-fitted is anisotropic (P3-low) yet should win; matrix is isotropic yet loses; jina-32d is low-dim yet loses.
- **Compared systems**: same 6 φ as B1.
- **Metrics**: scatter/rank-correlation of **retention (and e2e utility) vs each of {P2 k-NN-syn-precision, P1 effective-dim, P3 anisotropy-cos}**. Report Spearman ρ.
- **Setup**: reuse B1 outputs — no new runs, pure analysis.
- **Success criterion**: retention correlates with P2 (ρ high) and NOT with P1/P3 alone.
- **Failure interpretation**: if retention tracks P1 or P3 instead, the binding-property story changes.
- **Table/figure target**: Figure 1 (property-isolation scatter).

### Block 3: Mechanism-coupling / simplicity — rank-based vs radius candidate set  ·  Priority: NICE-TO-HAVE
- **Claim tested**: "P3's harm is sidesteppable by rank-based k-NN selection" (doc P3 note).
- **Why**: if a rank-based variant erases the matrix's disadvantage, isotropy is confirmed *moot* and the φ search relaxes to "maximise P2/P4, ignore P3" — an elegance result. Requires re-deriving/asserting ε-LDP for the variant.
- **Compared systems**: radius-based (current) vs rank-based `candidates`, each × {counter-fitted, matrix, qwen}.
- **Metrics**: retention, `|C_r|` stability, empty_%, + a note on the ε-LDP re-proof obligation.
- **Success criterion**: rank-based narrows the matrix↔counter-fitted retention gap.
- **Failure interpretation**: if not, P3 harm is intrinsic to the absolute-distance form regardless.
- **Table/figure target**: Appendix table.

### Block 4: Frontier necessity — neural Phrase-BERT vs static counter-fitted  ·  Priority: NICE-TO-HAVE
- **Claim tested**: is a GPU neural φ worth it over a CPU static one?
- **Why**: if static counter-fitted ties/beats Phrase-BERT, the elegant answer is "**no GPU model needed for φ**" — a strong simplicity claim. If Phrase-BERT wins on multi-word entities (P4/P6), that justifies the neural cost.
- **Compared systems**: counter-fitted vs Phrase-BERT, on single-token vocab AND on a multi-word-entity sub-eval.
- **Metrics**: retention, k-NN-syn-precision, OOV-drop, e2e on a proper-noun-heavy subset.
- **Success criterion**: a clear winner with a stated cost/benefit.
- **Table/figure target**: Appendix table.

### Block 5: Failure analysis — OOV-drop + e2e masking diagnosis  ·  Priority: NICE-TO-HAVE
- **Claim tested**: C3 scope.
- **Why**: quantify P6 (proper-noun drop) per φ, and test whether the extraction LLM washes out φ-retention differences e2e (the prior A/B hint).
- **Metrics**: OOV-drop rate on a proper-noun corpus; e2e utility delta vs mechanism-retention delta (does a big retention gap shrink to a small e2e gap?).
- **Table/figure target**: Appendix + honest-scope paragraph in the claim.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0 Sanity** | Build row-aligned φ caches; verify probes run | static loaders (counter-fitted, fastText, jina-32d via served, Phrase-BERT via HF); new probes self-check | all caches load, diagnostics non-degenerate on a 500-token toy | CPU + 1 GPU pass (Phrase-BERT ~12k words, jina if not served) ~mins | OOV: counter-fitted ~65k vocab → intersection w/ 12k cl100k tokens may be small → decide skip-perturb vs intersect-vocab |
| **M1 Baselines** | Re-confirm qwen3-emb + matrix panel | `eval.py` on existing 2 caches | numbers match `results/e2e_ab*.json` | reuses LLM cache, ~mins | none |
| **M2 Main** | Block 1 full panel A/B | `eval.py --caches <6> --attacks` + new probes | counter-fitted tops retention at matched `|C_r|` | geometry CPU; e2e = LLM calls on dev.txt (cached) | jina-v3 license (research-only); GPU one-at-a-time |
| **M3 Decision** | Blocks 2–4 | analysis (B2, no runs) + rank-variant (B3) + Phrase-BERT eval (B4) | P2 explains ranking; pick static-vs-neural | mostly CPU + 1 GPU | ε-LDP re-proof for B3 |
| **M4 Polish** | Block 5 + neighbour dumps | OOV-drop, e2e-masking, qualitative k-NN tables | — | CPU | — |

**Must-run**: M0, M1, M2 (Blocks 1–2). **Nice-to-have**: M3 (Blocks 3–4), M4 (Block 5).

## Compute and Data Budget
- **Total GPU**: minimal — only φ-cache builds that need a model (Phrase-BERT one pass over 12k words; jina-v3 if not served). Static caches (counter-fitted/fastText/GloVe) are CPU. Probes are numpy over 12k×{300..1024} (seconds). **One GPU process at a time** (Strix Halo gfx1151) — serialise the Phrase-BERT/jina builds.
- **Data prep**: download counter-fitted-vectors.txt (Apache-2.0), fastText cc.en.300; write a `build_from_static_vectors(words, path)` loader + a served/HF embed path for jina/Phrase-BERT (mirror existing `build_from_model_matrix`). Row-align all caches to `data/vocab.json` words.
- **Human eval**: none (k-NN-syn-precision via WordNet; optional LLM-judge spot-check).
- **Biggest bottleneck**: e2e LLM generation on dev.txt — already cached in `data/llm_cache`; keep corpus small.

## Risks and Mitigations
- **Vocab mismatch (P6)**: counter-fitted's ~65k word vocab vs 12k cl100k *subword* tokens — many cl100k tokens aren't whole words. *Mitigation*: evaluate on the intersection and report coverage; treat OOV as skip-perturb (no φ leak). Decide before M2.
- **e2e masking (C3)**: extraction LLM may erase φ differences. *Mitigation*: lead with mechanism-level metrics (retention, attack surface); frame e2e honestly per C3, don't claim e2e utility if it ties.
- **Matched-privacy validity**: `|C_r|` match alone may not equalise privacy. *Mitigation*: also report S_w, `token_channel_mi`, inv@K side-by-side; only compare retention where these are comparable.
- **B3 DP correctness**: rank-based variant departs from RANTEXT's radius derivation. *Mitigation*: assert/re-prove ε-LDP before reporting it as a method, not just a probe.
- **jina-v3 license**: CC-BY-NC → research-only; keep it as a control, not a shipped φ.

## Final Checklist
- [x] Main paper tables covered (Table 1 panel, Fig 1 isolation)
- [x] Novelty isolated (B2 — P2 vs P1 vs P3 correlation)
- [x] Simplicity defended (B4 static-vs-neural; B3 rank-based minimalism)
- [x] Frontier contribution justified-or-not (B4 decides if neural φ earns its GPU)
- [x] Nice-to-have separated from must-run (M2 must; M3/M4 optional)
