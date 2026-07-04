---
type: research
status: current
created: 2026-07-03
updated: 2026-07-03
tags: [issue, echo, absorption, gen-absent, remote-llm, fact-recall, prompting, rl-reward,
       round-trip, surrogate]
companion: [../plans/2026-07-03-surrogate-rl-gaps-fixes.md,
            ../specs/RL/surrogate-ranker-infiller.md,
            ../plans/2026-07-02-roundtrip-grpo-training.md]
---

# Issue: the remote LLM absorbs naturalistic fills (echo failure dominates realized utility)

**Measured (2026-07-03, `results/surrogate_env_diagnostics.json`, 16 docs × tau_walk/all_floor ×
3 corpora on the persisted arms artifact):** of the unique generalization fills in `doc_p`,
the remote model (`Qwen3.6-35B-A3B`) reproduces verbatim only 0–7%; **82% (clinical) / 94%
(enron) / 95% (aeslc) are absorbed** — no surface trace in `out_p` at all (fuzzy < 70 *and*
embedding sim < 0.6), so no extractor, rule-based or learned, can restore the original surface
into `out_final`. The loosely-echoed remainder (<10% everywhere, and partly spurious fuzzy hits
on note section headers) is why the E1 semantic-aligner was descoped
([gaps plan, Phase M](../plans/2026-07-03-surrogate-rl-gaps-fixes.md)). Placeholder caveat
(corrected 2026-07-04): `ph_swapped 9/9, zero residue` measures *inversion given echo* — every
placeholder token that appeared in out_p swapped back cleanly. It does **not** mean placeholders
reliably appear: sampled docs show 1/3 and 0/4 placeholder tokens echoed, vs prose fills the
task needed ("the autumn") echoing fine. **Echo is dominated by task relevance, not fill form.**

## Definitions

- **Echo** — the remote model reproduces a fill's surface (verbatim or loosely) in `out_p`,
  giving the extractor an anchor to invert.
- **Absorption** — the model uses (or drops) the fill's information without any surface trace in
  `out_p`. Unrecoverable downstream by construction.
- **Task omission** — the output legitimately does not restate a span because the task doesn't
  need it (a subject line restates almost nothing). Not a defect.
- **Substitution-induced absorption** — the part of absorption *caused by the mechanism*: the
  fact would have been restated from `doc_orig` but is not restated from `doc_p`.
- Other terms (fill modes, ŝ, u_qa, fact recall): [RL spec](../specs/RL/surrogate-ranker-infiller.md#definitions).

## Scope — how much of this is actually harm

Two corrections keep the headline numbers honest:

1. **Raw absorption conflates task omission with mechanism damage.** Even the unperturbed
   round trip restates only a fraction of probed facts (no_privacy fact recall 0.13 clinical /
   0.18 enron — task style, `max_tokens` truncation, reader strictness). The mechanism's own
   cost is the *delta*: no_privacy 0.13–0.18 → tau_walk 0.01–0.04 on realized fact recall. That
   delta — roughly a 5–10× drop on the facts the task provably restates — is the harm; the
   82–95% raw absorption rate overstates it because most absorbed fills were never going to be
   restated by any output.
2. **For some tasks, dropping substituted spans from `out_p` is the intended behavior** (a
   subject line or summary that omits PII-adjacent detail is fine, sometimes preferable). The
   headline utility (fact recall on `out_final`) already encodes this scoping: probes exist only
   for **gold-restated** spans (R ∩ gold), so absorption of task-irrelevant fills costs nothing
   by construction. The issue is precisely and only: *gold-restated facts whose fills the model
   absorbs*.

Refinement measurement (open): absorption rate restricted to probe-bearing surfaces, per corpus
— the exact harm rate. Current probe supply (3.2/1.0/0.4 per doc) makes this a small-n cross-tab;
compute it when probe supply grows.

## Mitigations

1. **Anchor-mode fills (primary, in the action space).** Placeholders echo; prose doesn't. The
   RL action spectrum (generic placeholder → descriptive placeholder → relational → naturalistic,
   [spec §2.2](../specs/RL/surrogate-ranker-infiller.md)) exists exactly to let the policy pay
   for echo-ability where the task needs the fact carried, and the ŝ echo-survival prior makes
   that trade visible to the reward.
2. **Stricter prompting (partial, cheap, testable).** Instruct the remote model to reuse marked
   input phrasing — e.g. "when your output references dates, names, organizations or quantities
   from the input, reuse the input's exact wording", or extend template-token treatment to
   naturalistic fills by bracketing them. Plausibly recovers part of the 5–10× delta given that
   the `<…>` syntax already induces copying. Costs and caveats: the prompt becomes **part of the
   mechanism** — it must be held fixed across all arms/methods (a per-method prompt is a
   calibration knob, banned), it may degrade generation naturalness (measure, don't assume), and
   it shifts every cached round trip (new environment version; re-measure ŝ, re-run the gate).
   Status: **untested — cheapest next experiment on this issue** (one prompt variant × the
   arms artifact ≈ 200 cached-forever remote calls).
3. **Not mitigations:** better extraction (absorption leaves nothing to extract — measured
   ceiling < 10%); lowering inversion thresholds (buys the spurious part of the loose bin and
   coincidence-echo false positives).

## RL-side implications

- **DECISION 2026-07-04: the ŝ prior is dropped; the reward deliberately does not price echo.**
  Two grounds: (a) ŝ re-binds the detached surrogate to one remote model's behavior; (b) the
  corrected evidence above says echo is dominated by task relevance — outside the ranker's and
  infiller's control — and an uncontrollable effect does not belong in the policy's reward. The
  echo channel is measured only at evaluation (fact recall on `out_final`); a
  surrogate-vs-realized gap dominated by policy-controllable echo effects triggers the
  round-trip reward upgrade, not an ŝ revival. Spec §5.2 carries the normative statement.
- **The surrogate reward cannot see any of this — accepted.** u_qa reads `doc_p`, never
  `out_p`; echo failure is in its null space
  ([gaps plan, Gap 2](../plans/2026-07-03-surrogate-rl-gaps-fixes.md)), fail-closed for
  placeholder carriage.
- **This is the strongest argument yet for the round-trip reward**
  ([2026-07-02-roundtrip-grpo-training.md](../plans/2026-07-02-roundtrip-grpo-training.md)):
  a round-trip reward observes the real `out_p` per candidate and therefore prices echo and
  absorption *directly, per fill* — no ŝ table, no mode coarseness, and the infiller could learn
  fill-level echo-craft (phrasings the model repeats), which the surrogate structurally cannot
  teach. The upgrade trigger in the surrogate plan ("error analysis shows the surrogate's blind
  spot dominating residual utility loss") is now *pre-confirmed at the baseline*: the blind
  spot's channel is measured as the dominant loss today. The surrogate path stays first (cost,
  model-unbinding), but if the ŝ-discounted reward fails the Phase-5 re-gate — i.e. mode-level
  discounting is too coarse to rank candidates — the round-trip reward is the designated
  fallback, not another surrogate iteration.
- **Prompt mitigation interacts with training:** any prompt change alters echo rates → new ŝ
  table → new reward → re-gate. Decide the prompt *before* training (it is an environment
  constant, like the extractor version).
- **Evaluation stays two-sided:** absorption also destroys *leak-through* — an absorbed fill
  can't leak the original in `out_final` either. Report utility and attack on `out_final`
  together; a mechanism that "wins" utility via anchors must show the anchors don't widen the
  `out_final` leak channel (placeholder swap-back reintroduces the original surface into
  `out_final` by design — that is the utility working, but it means `out_final` privacy rests
  entirely on `doc_p`-side protection having been sufficient).

## Sources

Measurements: `results/surrogate_env_diagnostics.json` (Phase M, arms artifact
`data/task_arms_tau0.02.json`); gate numbers `results/surrogate_validation.json`. Plans:
[gaps + fixes](../plans/2026-07-03-surrogate-rl-gaps-fixes.md),
[surrogate GRPO](../plans/2026-07-02-surrogate-grpo-training.md),
[round-trip GRPO](../plans/2026-07-02-roundtrip-grpo-training.md). Anti-extractor deletion
lesson: [NaPaRe](../../research-wiki/papers/huang2025_tree_search_rewriting.md)
([arXiv 2509.20838](https://arxiv.org/abs/2509.20838)).
