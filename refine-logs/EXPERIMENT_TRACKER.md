# Experiment Tracker — Lever 2 φ

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | build static φ caches, row-aligned | counter-fitted, glove (same vocab/dim) | 7913 sub-vocab | coverage 66% | MUST | **DONE** | `build_from_static_vectors`; glove = P2-isolation control (cf's pre-retrofit base). fastText deferred |
| R002 | M0 | build neural φ cache | Phrase-BERT (CPU, 768d) | 7913 sub-vocab | Δφ=0.365 | MUST | **DONE** | ran on host CPU (110M) — no GPU contention. jina low-dim deferred (v3 not served; v5-omni/jina-code are) |
| R003 | M0 | new probes | syn_precision (WordNet), eff_dim, ret_corr | sub-vocab | run clean | MUST | **DONE** | in diagnostics.py + phi_geometry_ab.py. rank-variant + OOV-drop deferred |
| R004 | M1 | baselines | qwen3-emb sub-vocab | sub-vocab | panel | MUST | **DONE** | re-embedded sub-vocab; in the A/B table |
| R005 | M2 | **geometry/mechanism A/B (Block 1)** | qwen, glove, cf, Phrase-BERT | sub-vocab, matched \|C_r\|=0.05 | rel_spread, eff_dim, aniso, retention, **ret_corr, syn_prec** | MUST | **DONE** | `results/phi_geometry_ab.json`. e2e (LLM) arm still TODO |
| R005e | M2 | e2e A/B arm (Block 1, C3) | all 4 φ | dev.txt (n=5) | utility, leakage, pii | MUST | **DONE** | `results/e2e_ab_subvocab.json`; **tied — extraction LLM masks φ**. n=5/seed=0, directional |
| R006 | M3 | property isolation (Block 2) | reuse R005 | — | ret_corr tied; syn_prec isolates cf | MUST | **DONE** | key finding: raw retention = anisotropy artifact; only counter-fitting moves floor-free P2 |
| R007 | M3 | rank-based mechanism variant (Block 3) | radius vs rank × {counter-fitted, matrix, qwen} | 12k | retention, empty%, ε-LDP note | NICE | TODO | re-prove/assert ε-LDP first |
| R008 | M3 | neural-vs-static (Block 4) | counter-fitted vs Phrase-BERT | tokens + multiword entities | retention, P2-prec, OOV, e2e | NICE | TODO | decides if GPU φ earns its cost |
| R009 | M4 | OOV-drop + e2e-masking (Block 5) | all 6 φ | proper-noun corpus | OOV-drop%, Δretention vs Δe2e | NICE | TODO | supports C3 honest scope |
