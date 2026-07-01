# Research Findings

## 2026-07-01 — Does *anisotropy* cause the coherence↔leakage trade? (result-to-claim)

**Claim tested (causal):** lower embedding anisotropy *causes* more coherent `Doc_p` +
higher perturbation-utility but *also* higher leakage/attackability (`inv@10`, `pii_leak`),
a coherence↔privacy trade governed by anisotropy, at matched `|C_r|`.

**Verdict: NO** (causal) / **partial** (descriptive). Codex, xhigh, high confidence.
Integrity: *provisional — no EXPERIMENT_AUDIT.json run.*

**What the data supports (descriptive only):** in the 2-embedding N=60 sweep, the
lower-anisotropy pythia-410m (aniso 0.011) had higher `coherence_gen_p`, `inv@10`, and
`pii_leak` than ada-002 (aniso 0.79) at every ε; `token_MI` and final utility ~tied.
Signs are consistent with a coherence/privacy tradeoff. (Final *utility* is essentially
equal, so "higher utility" is NOT supported — drop that half of the claim.)

**Why the causal attribution fails — confounds not separable in a 2-point comparison:**
1. **Vocab coverage/size** — ada |V|=10129 vs pythia 12000, different token sets. ada drops
   common cased names (Sarah/Johnson/Chicago) → privacy-by-*removal* mechanically lowers
   ada's `pii_leak` and coherence (holes), independent of geometry.
2. **`|C_r|` not actually matched** — realized ada/pythia ratio 0.67→2.0 across ε (at ε=10
   ada=1.4% vs pythia=0.7%). The "matched operating point" premise is violated.
3. **eff_dim differs 5×** (116 vs 570) — a separate geometry axis (concentration/rankability),
   not anisotropy.
4. **Bundled system differences** — contrastive text-embedder vs LM `embed_tokens`, dim
   1536 vs 1024, different tokenizer/coverage. Anisotropy is one of many differences at once.

**Plausible single mechanism (hypothesis, unproven here):** removing the common direction
raises distance/rank *contrast* among neighbors → better semantic neighbor selection
(↑coherence) AND a sharper neighborhood constraint on the original token (↑inversion). The
current data cannot separate this from confounds 1–4.

**Next experiment (minimal clean control, caches already built):** e2e sweep on
`data/vocab` (qwen3-emb RAW, aniso 0.61) vs `data/vocab_centered` (SAME matrix mean-centered,
aniso ≈0) — identical V/token-set/dim/tokenizer/docs/ε/pipeline/attacker, **matched realized
`|C_r|` (match the distribution, not just the mean)**, paired seeds, CIs over docs. Add a
dose-response (subtract α·mean for several α) to test whether effects track anisotropy
continuously. This isolates anisotropy *within one embedding*; it does NOT prove anisotropy
governs the trade across arbitrary embedding families.

**Do not, until that runs:** state anisotropy as the cause in `embedding-map.md`/the HTML.
Current wording ("governed by anisotropy") must be softened to "correlated with anisotropy;
confounded by vocab coverage and unmatched |C_r|". Related wiki claim:
`research-wiki/claims/anisotropy-bad-but-insufficient.md` (status untouched — proof axis).

## 2026-07-01 — Anisotropy mechanism: CONTROLLED TEST → REFUTED

Ran the clean control the gate demanded: qwen RAW (aniso 0.613) vs SAME matrix mean-centered
(aniso 0.000) — identical vocab (12000)/dim/tokenizer/docs/seed/pipeline/attacker, realized
|C_r| matched ~1.3% at ε=2/6/10 (ε=14 unmatched, discounted). N=10, single seed, MAUVE off.
`results/dp_sweep_qwen_{raw,centered}.json`.

**Result (Δ = centered−raw, matched ε):** coherence Δ mixed (−0.03/−0.11/+0.01, ≈0 net);
inv@10 Δ = −0.20/−0.24/−0.14 (centering LOWERS inversion); pii_leak Δ = −0.07/−0.13/−0.20
(LOWERS); utility unchanged.

**Verdict: mechanism REFUTED.** Isolating anisotropy does not reproduce the ada-vs-pythia
trade; the leakage signs REVERSE (ada-vs-pythia: isotropic→higher inv@10/pii; controlled:
isotropic→lower). So the ada-vs-pythia "coherence↔attackability trade" was a confound (vocab
coverage/name-drop, eff_dim, embedding family), NOT anisotropy — exactly as the result-to-claim
gate warned. Also refutes the older "anisotropy weakens the inversion attack" reading.

**Standing conclusion:** φ's only established effect is the radius-calibration constraint
(§2/effect-1). Anisotropy matters only because it sets the distance scale the radius must match;
it is not a privacy↔coherence dial. At matched |C_r|, centering is weakly privacy-positive
(↓inv@10, ↓pii, flat coherence/utility) — needs N≥30 + seeds + dose-response to firm.

Docs updated: embedding-map.md §3a (verdict) + bottom line + §3; HTML §5.2 conclusion.
