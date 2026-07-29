---
type: plan
status: current
created: 2026-07-29
updated: 2026-07-29
tags: [ranker-v2, html, figures, controller, documentation]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md]
---

# Plan — update interactive-ranker-v2.html figures to the current implementation and spec

Target page: `docs/html/interactive-ranker-v2.html` (renderer `docs/html/js/ranker-model-diagram.js`,
styles `docs/html/css/site.css`). Focus figures: **M1 (semantic policy architecture)**,
**T2 (training process and reward flow)**, **I1 (deployed document inference)**; T1 gets one
small secondary touch. The page was reviewed rendered (headless Chromium screenshots) on
2026-07-29; all edits below are relative to that state.

## Ground truth — read these first

- `docs/specs/RL/interactive-ranker-v2.md`, especially the new section
  **"Implementation status and interim deviations (2026-07-29)"** at the end — it enumerates
  exactly the deltas the figures must absorb.
- `docs/specs/RL/interactive-ranker-v2-decision-log.md`, entry
  **"OPEN FORK — controller strength: switch-calibrated, gap-scaled alpha (2026-07-28)"** —
  the controller changes below come from its adjudication.
- Code: `src/cloak/ranker/semantic.py` — `distribution()` (controller math, gap scaling,
  lambda-zero branch) and `switch_threshold_calibration` / `calibrate_alpha` (alpha init);
  `scripts/train_interactive_ranker.py` — train flags `--alpha-utility-routing`,
  `--controller-gap-scaling`, `--alpha-init` and the KL-reference capture order;
  `src/cloak/qa/scoring.py` — `assertion_reward_role` (policy vs monitoring roles,
  policy-only denominator).

## Standing directives (do not deviate)

1. **Keep the semantic privacy head in every figure.** The live trainer currently injects
   direct grounded counts (`DirectCountPrivacyProvider`) instead of the learned head, but this
   is a *temporary interim* while a proper k-anonymity estimator model is trained. Timo's
   explicit decision: diagrams depict the privacy head, not direct-count injection. Do not add
   direct-count nodes, do not relabel `p_hat` as a count lookup. At most, one sentence of page
   prose may note the interim and point at the spec's deviations section.
2. **Controller changes are an adopted candidate, not a closed decision.** Switch-calibrated
   alpha init + gap scaling passed all responsiveness criteria; final adoption is gated on a
   production-trainer re-evaluation (decision log, fork status). Depict them as the production
   controller configuration and let the caption/prose reference the decision-log fork; do not
   present them as unconditionally final.
3. No plan-phase/requirement identifiers in node labels or ids — name things after what they
   are (repo naming rule).
4. M1 is guarded by a DOM contract: `scripts/spikes/check_ranker_m1_dom_layout.mjs` (run with
   `node`). It pins node ids, HTML-node ownership, and SVG outline conventions. Keep it passing;
   extend its node-id list if you add nodes.

## FIG · M1 — Ranker v2 semantic policy architecture (`data-module-id="controller"` region)

Current controller chain (nodes `additive_controller`, `alpha`, `lambda_transform`,
`privacy_control`) draws `b(a,λ) = α · g(λ) · p̂(a)` added onto `u(a)`. Changes:

1. **Add the gap-scale factor to the multiplication chain**: the controller is multiplied by
   the *detached per-menu utility-logit range* — `(max_a u(a) − min_a u(a))`, stop-gradient —
   so a sharpening utility tower cannot silently defeat the controller. New small operator node
   (suggested id `utility_gap`, label like "utility-logit range · detached") fed from the
   `utility_logit` tensor node, entering the product alongside `alpha` and `g(λ)`. Update the
   module header formula to `b(a,λ) = α · g(λ) · gap · p̂(a)`.
2. **Annotate alpha initialization**: `alpha` node (or its hover/subtitle) gains
   "init = weighted-median switch threshold measured on BC menus (gap-normalized)". Source:
   `switch_threshold_calibration` in `semantic.py`.
3. **Annotate the exact lambda-zero identity** on the controller module: at `λ = 0` the code
   takes a separate branch — combined logits are *identically* the utility logits, not merely
   `b = 0` numerically. One short label/caption line.
4. **Prose updates in the same section**:
   - "Controller parameters." paragraph (near line 438): alpha is no longer described as a
     plain trainable scalar starting at its default; add calibrated init + gap scaling
     (candidate, decision-log pointer).
   - The `z(a, lambda)` equation block and the `g(lambda)` table rows (near lines 698–710),
     including the pairwise log-odds line `log pi(a)/pi(b) = …` — all gain the `gap` factor.

## FIG · T2 — Training process and reward flow

Panel **01 · freeze the environment**:

- "QA assertion artifact" node: subtitle `weights · dependencies · Z_d` → reflect the
  policy-role denominator (`qa-utility-runtime-v2`): assertions carry policy vs monitoring
  roles; `Z_d` is policy-role weight mass only; monitoring (document-wide demographic probes,
  exact-relation contracts) is scored and reported but never credited.
- Add the two load-time training transforms (either a new node or lines on an existing one):
  **initial-RL scope** — only `drug`, `health-condition`, `medical-procedure` are
  ranker-controlled; out-of-scope or count-uncovered decisions are demoted to fixed KEEP —
  and **zero-signal document drop** — documents with no policy reward mass are excluded from
  training.

Panel **02 · build a semantic starting point**:

- "calibration + menu" box: name what the calibration preflight actually freezes — the lambda
  menu and the threshold manifest, with measured (never invented) quantities: counterfactual
  probe nonzero rate, reader-jitter non-inferiority margin.
- **Add a controller-calibration step** after BC/ExIt: switch thresholds measured on the
  warm-started policy's menus → alpha initialized (gap-normalized median) → **KL reference
  captured from the calibrated policy** (anchoring KL to an uncalibrated reference would pull
  the controller back to the dead regime). This ordering is implemented in
  `scripts/train_interactive_ranker.py`.

Panel **03 · lambda-conditioned hybrid episode**:

- "provisional utility credit" node: `fixed Z_d · no SD normalization` → `fixed policy-role
  Z_d · no SD normalization`; add "monitoring reported, never credited".
- "exact count objective" node and the "Exact count view." prose (near line 917): the count
  view formula gains the same detached gap factor:
  `pi_count = softmax(mask(stopgrad(u) + alpha · g(lambda) · gap · stopgrad(p_hat)))`.
  "gradient → alpha only" stays true and stays in the node.
- "joint optimization" node: keep; optionally note KL regularizes against the *calibrated*
  reference.
- Prose near lines 881/905: wherever `Z_d` is described as the fixed denominator, qualify it
  as the fixed **policy-role** denominator.

## FIG · I1 — Deployed document inference

- "semantic-v1 forward" node (near line 1021): `u + alpha·g(lambda)·p_hat` →
  `u + alpha·g(lambda)·gap·p_hat` (same gap factor as M1; keep it compact).
- "detect + group" node: add the current controlled-type scope — controlled types (initial
  scope: drug · health-condition · medical-procedure) become policy decisions; PERSON/CODE are
  rule-substituted placeholders (consistent with "claimed fills = fixed claims" in
  "begin document"); other detected values are fixed KEEP, outside the action space.
- "local unperturb" node (optional, one line): the substitution record restores alias groups —
  every fill in an alias group inverts to the group's first source surface (deterministic
  inversion, recently fixed defect).
- Everything else in I1 verified accurate against the implementation on 2026-07-29 (local
  ranking loop, argmax deployment, remote boundary, "absent at inference" list).

## FIG · T1 — secondary touch only

"alpha ≥ 0 · one global scale" node and the logit-sum: add the gap factor to the red
controller path and one caption word on calibrated init. The dashed-supervision caption
("exact counts update alpha only during hybrid RL") remains correct. Nothing else.

## What was checked and needs no change

- M1 dimensions and modules (encoder, pair 3840, fusion, three attentions, memory KV 3072 /
  query 1536, utility MLP 4608→768→1) match `semantic.py` as of commit `0dafd2d`.
- No stale model names in the page (remote/reader are described generically; the encoder pin
  `thomas-sounack/BioClinical-ModernBERT-base@c3648aa8…` is current). Remote task model and QA
  reader are both `medgemma-4b-it` if a name is ever added.
- T2's freeze/starting-point/episode topology and I1's remote boundary match the spec.

## Verification before finishing

1. `node scripts/spikes/check_ranker_m1_dom_layout.mjs` passes (extend its pinned node-id list
   for any added node).
2. Screenshot the four figures rendered and eyeball them. Recipe: playwright's chromium (a
   local install exists under `~/.hermes/hermes-agent/node_modules/playwright`; browser
   binaries in `~/.cache/ms-playwright`), `file://` load of the page, wait ~2.5 s for the
   renderers, then before element screenshots widen every element whose
   `scrollWidth > clientWidth` to `max-content` with `overflow: visible` — otherwise the wide
   diagrams clip to their scroll containers.
3. Cross-read the edited captions against the spec's "Implementation status and interim
   deviations (2026-07-29)" section — figure text must not promote the controller candidate to
   a closed decision and must not surface direct counts.
