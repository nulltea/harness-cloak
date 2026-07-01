---
type: experiment
node_id: exp:lever2-phi-model-ab
title: "Lever 2 φ-model A/B: synonym structure vs isotropy vs dynamic-range vs e2e"
idea_id: "idea:llm-matrix-phi"
verdict: partial
confidence: medium
date: 2026-06-30
hardware: "host .venv: CPU (numpy geometry) + ROCm GPU (SimCSE e2e); served qwen3-emb + jina-v5"
duration: ""
provenance: "scripts/phi_geometry_ab.py + results/phi_geometry_ab.json; scripts/phi_e2e_ci.py + results/phi_e2e_ci.json; docs/research/embedding-map.md; review-stage/AUTO_REVIEW.md (auto-review 3→7/10); result-to-claim verdict partial"
tags: ["lever2", "phi", "anisotropy", "synonym-structure", "counter-fitted", "jina", "ab"]
---

# Lever 2 φ-model A/B

**verdict:** `partial`  ·  **confidence:** `medium`  ·  supports `claim:anisotropy-bad-but-insufficient` (revised)

## Metrics
Whole-word sub-vocab (7,913 tok), matched candidate load |C_r|≈0.05, ε=3, 8 seeds, bootstrap CIs.
syn_prec@10 on fixed 7,486-query set, paired vs glove:
- **C2 (yes, caveats):** cf +0.061 [.058,.063] ≫ Phrase-BERT +0.013 [.010,.015] > qwen +0.003 ≈ whitened-glove +0.002; jina-v5 −0.067 / jina-64d −0.079 (worst). Only counter-fitting moves P2 materially.
- **C1 (partial):** raw retention floor-confounded; corrected (cos−floor)/(1−floor) collapses cross-model spread 0.36→~0.02 (qwen/glove/cf/phrasebert 0.43–0.45). NOT a formal tie (margin 0.076; 0.622 incl. jina).
- **C1b/P3 (yes, this control):** whitened-glove (aniso≈0) lowers ret_corr 0.446→0.377, syn_prec flat.
- **P1 (yes, negative control):** jina-v5 max rel_spread 0.45–0.51, eff_dim 7.9→3.4, ret_corr≈1.0 (degenerate) yet worst syn_prec → dynamic-range/low-dim alone ≠ synonym structure.
- **C3 (partial):** e2e qwen vs cf (25 pairs, remote Gen_p + final + PII) — all Δ straddle 0; pilot, 5 docs, not generalizable.

## Reasoning
P2 (local synonym structure), reachable only by explicit synonym-fitting, is the φ property that matters
for the perturbation geometry; isotropy (P3) and dynamic-range/low-dim (P1) are each insufficient and the
controls maximising them score worst on P2. Raw retention is an anisotropy-floor artifact and a degenerate
metric on rank-collapsed φ. But the P2 edge does not translate end-to-end at ε=3 (pilot) → **Lever 2 has low
e2e leverage at this operating point.** Inference caveats: WordNet syn_prec CIs likely too narrow (correlated
queries → cluster bootstrap pending); C3 underpowered.

## Connections
supports → `claim:anisotropy-bad-but-insufficient` (revised: P2 binds; retention is a floor artifact).
Controls whitened-glove (P3) and jina-v5 (P1) both isolate that neither isotropy nor dynamic-range delivers P2.
Edges recorded in `graph/edges.jsonl`.
