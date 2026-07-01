# Auto Review Log — Lever 2: the embedding map φ

Topic: "Lever 2 φ — retention is an anisotropy artifact; φ choice is masked e2e."
Reviewer: Codex / gpt-5.5, xhigh. Difficulty: medium. Backend: codex. threadId 019f1534-…
Artifacts reviewed: docs/research/embedding-map.md, refine-logs/EXPERIMENT_{PLAN,RESULTS}.md,
results/phi_geometry_ab.json, results/phi_e2e_ci.json, research-wiki/claims/anisotropy-bad-but-insufficient.md,
scripts/phi_geometry_ab.py, scripts/phi_e2e_ci.py, src/inferdpt/{diagnostics,embeddings,probes/_common}.py.

## Score progression
| Round | Score | Verdict |
|---|---|---|
| 1 | 3/10 | Almost (needs cleanup) |
| 2 | **7/10** | **Almost → STOP** (≥6 ∧ almost) |

## Round 1 (3/10, Almost)

### Assessment
Verified reported numbers against the JSON. Geometry correction + counter-fitting isolation are good,
but: C3 underpowered (n=5, single seed) and partly architecturally baked (extractor sees raw Doc);
"matched |C_r|" ≠ "matched privacy"; the corrected-retention "tie" had no CIs/equivalence margin;
C2 isolation lacked an isotropy-only control; WordNet syn_prec is narrow.

<details><summary>Round 1 raw reviewer response</summary>

Score: 3/10. C1 partly supported (correction formula sound as descriptive normalization, but "tie" not
formally established — no CIs/equivalence test). C2 supported for WordNet metric — independent recompute:
glove 0.0985, cf 0.1592, paired diff +0.0607, 95% CI ~[+0.058,+0.063]; "only" too broad (fastText/jina/
whitening/rank not tested). C3 not supported at claimed strength — n=5, one seed, extractor given raw Doc
so φ-insensitivity structurally expected; supports "pilot shows no obvious φ effect", not a settled claim.
Weaknesses: (1) C3 underpowered+baked → reframe pilot / many docs+seeds, bootstrap paired CIs, report
remote-only Gen_p separately; (2) matched |C_r| ≠ matched privacy → rename, report S_w/MI/inv/entropy;
(3) corrected-retention tie not an equivalence result → bootstrap CIs + explicit margin; (4) C2 isolation
incomplete → add whitened/all-but-top GloVe, fastText, low-dim; (5) WordNet narrow → add LLM/human judge
+ NN tables; (6) sub-vocab 66% external validity. Readiness: Almost for an honest internal short report.
</details>

### Actions taken
Bootstrap CIs + equivalence margin on corrected retention; fixed shared 7,486-query paired syn_prec;
whitened-GloVe isotropy-only control; multi-seed paired e2e separating remote Gen_p from final output;
SimCSE scorer moved to GPU; reframed "matched candidate load". (Also: cleaned cross-project ROCm
launcher/Containerfile, enabled disk LLM cache, scoped e2e to the decisive qwen-vs-cf pair.)

## Round 2 (7/10, Almost — STOP)

### Assessment
Geometry side much stronger; verified updated JSON. Blockers now reporting/aggregation correctness, not
the core result. C1/C2 credible as internal finding; C3 credible only as a pilot at ε=3.

<details><summary>Round 2 raw reviewer response</summary>

Score: 7/10. Verdict Almost — ready after a small cleanup pass. Matched-privacy language addressed
(suggest also storing achieved |C_r|/V — recomputed ~0.042–0.047). Equivalence/CIs mostly addressed but
ret_corr CI is seed-mean percentiles not a bootstrap → use boot_ci or relabel. Isotropy control addressed
(whitened GloVe) — supports "this whitening retrofit hurts", not "isotropy always counterproductive".
C3 improved (remote vs final separation) but 25 pairs over 5 docs → "properly powered/confirmed" too
strong. Weaknesses: (1) PII reporting wrong — JSON has bare NaN (invalid) and docs say "no PII" but 15/25
rows have finite pii_leak; recomputed qwen 0.523, cf 0.559, paired +0.036 CI ~[−0.011,0.081] → use nanmean,
emit null, say "partially observed, inconclusive"; (2) C3 still a pilot generalization — cluster-bootstrap
by doc / add docs; (3) wording too broad — "only counter-fitting moves P2 materially", "this isotropy-only
whitening control"; (4) stale claim file still says "tied" + old n=5 evidence → update/supersede; (5) raw
retention discussed but not stored in geometry JSON → add for auditability. Bottom line: fix PII/NaN,
ret-CI labeling, stale wording → ready.
</details>

### Actions taken (termination cleanup)
1. **PII bug** — `phi_e2e_ci.py` now `nanmean` + JSON `null` for missing rows + paired pii_leak CI;
   corrected docs: PII observed in 15/25 rows, cf−qwen +0.036 [−0.011,+0.081] (inconclusive). My earlier
   "Presidio found no PII" was a `nan`-propagation artifact and is now fixed.
2. ret_corr CI now a bootstrap (`ret_corr_CI_boot`); raw retention + achieved |C_r| stored in the JSON.
3. Wording narrowed; stale claim Statement superseded with a floor-corrected revision.

### Results (final, after cleanup)
- Geometry (`results/phi_geometry_ab.json`, matched load 0.042–0.047): corrected retention qwen 0.453
  [.447,.46], glove 0.446, **whitened-GloVe 0.377** [.371,.384] (outlier; raw==corrected, floor≈0),
  cf 0.431, Phrase-BERT 0.434. Equivalence margin 0.076 (not within 0.03). syn_prec paired vs glove:
  cf +0.061 [.058,.063] ≫ Phrase-BERT +0.013 > qwen +0.003 ≈ whitened-GloVe +0.002.
- e2e (`results/phi_e2e_ci.json`, 25 pairs): cf vs qwen — remote −0.019 [−0.066,+0.024], final −0.008
  [−0.036,+0.019], pii +0.036 [−0.011,+0.081]. All straddle 0.

## Method Description
RANTEXT (InferDPT) perturbs each token by sampling from a random-radius Euclidean ball in an embedding
space φ, scored by an exponential mechanism `u=1−d/radius`. This study evaluates φ-model fitness via
(a) geometry/mechanism probes on a 7,913-word whole-word sub-vocab at matched candidate load
(rel_spread, eff_dim, anisotropy floor, anisotropy-corrected retention, WordNet synonym precision on a
fixed query set), and (b) a multi-seed paired end-to-end pipeline (perturb → remote gen `Gen_p` → trusted
extract) that separates the φ-sensitive remote stage from the re-grounded final output. Candidates:
qwen3-embedding-0.6B (baseline), GloVe, counter-fitted (synonym retrofit of the same GloVe base),
whitened-GloVe (isotropy-only control), Phrase-BERT. Heavy torch scoring (SimCSE) runs on the AMD iGPU
on the host `.venv` GPU; geometry is pure-numpy. Finding: raw retention is an anisotropy artifact;
only the counter-fitting retrofit materially improves floor-free synonym precision; that edge does not
reach end-to-end at ε=3 (pilot) → φ (Lever 2) has low end-to-end leverage at this operating point.

## Status: COMPLETED (stop condition met at round 2: 7/10, Almost)
Remaining (deferred, optional): cluster-bootstrap e2e over more docs to lift C3 above pilot; jina low-dim
+ fastText + rank-based arms; LLM-judge synonym precision.
