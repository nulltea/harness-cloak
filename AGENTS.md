# HarnessCloak — privacy layer for closed-box LLM inference

## Goal
A **privacy layer for closed-box (remote) LLM inference**: locally rewrite a prompt `doc_orig` →
anonymized `doc_p`, send `doc_p` to the remote LLM, receive `out_p`, then locally **un-perturb**
`out_p` → `out_final`. Hide the sensitive content of `doc_orig` from the remote model while preserving
the utility of `out_final`.

**Legacy (abandoned): InferDPT + RANTEXT.** A single global/scalar ε budget applied per token ruins the
privacy↔utility tradeoff — one knob is forced to govern identity-removal, premise-preservation, and
extraction-signal at once. Write-up: report `docs/html/infer-dtp.html`, taxonomy
`docs/research/rantext-limitations.md`.

**Current direction:**
1. **Learned context-aware substitution** for the rewrite (`doc_orig`→`doc_p`), replacing RANTEXT's
   per-token metric-LDP token swaps — see `docs/research/learned-substitution.md`.
2. **An efficient, tailored model architecture** for extraction (`out_p`→`out_final`), replacing the naive
   general-LLM extractor — a denoise/edit reconstructor, local, zero DP cost by post-processing immunity.

**Positioning:** the pipeline must be **tailorable to specific user needs** — user-specified sensitive
types and their generalization lattices, not only fixed schemas. The span detector is therefore a
composition (supervised fixed-schema core + a cheap per-user-type path: gazetteer/zero-shot/fine-tune);
zero-shot and fine-tune extensibility are first-class requirements. See
`docs/research/learned-PII-detection.md`.

**What matters (the only comparison that counts):** privacy vs utility at **matched realized privacy** —
privacy measured against an **LLM re-identification attacker**, utility measured on `out_final`.
Everything else (embedding geometry, lexical overlap, `|C_r|`) is a diagnostic, never the basis of a
method comparison.

## Empirical honesty (hard rule — do not break)
- Compare methods (substitutors, reconstructors, mechanisms) **only at matched realized privacy and
  identical settings**. **Never** invent or implicitly apply a per-model calibration/normalization knob to
  equalize a secondary quantity — it places methods at *different real privacy levels* and invalidates the
  privacy↔utility comparison. This already produced a wrong conclusion once (RANTEXT `noise_scale`).
- **Measure privacy against an adversary, not a surface metric.** Report privacy as the success of an LLM
  re-identification / inference attacker on `doc_p` (and `out_final`); n-gram overlap, self-substitution,
  and embedding distance overstate protection. Watch the substitutor's own memorization as a leak channel.
- Measure privacy/attack/utility as **outcomes**. If a method degenerates at fixed settings, **that is the
  finding — report it, don't engineer around it.**
- Move operating points only with a method's legitimate privacy knob (ε for DP methods; the privacy target
  otherwise); compare across operating points with a Pareto curve at equal *realized* privacy — never a
  per-model fudge factor.
- No result claims without the run output; state degeneracies, confounds, and caveats plainly.

## GPU — run heavy workflows in the host `.venv`

The host `.venv` **runs on the GPU directly** — no container. Use it for **all** heavy workflows —
rewriter/reconstructor training + inference, leakage/MI/utility probes, re-identification attacks, and
`pytest`:

```bash
# sanity: .venv/bin/python -c 'import torch; print(torch.cuda.is_available())'   # -> True
```

One AMD Strix Halo iGPU (gfx1151). **One GPU process at a time**: don't launch a second GPU run while
one is live; wait on long runs, never poll-spin.

**Always run long/background jobs unbuffered** — `.venv/bin/python -u …` (or `PYTHONUNBUFFERED=1`). Python
block-buffers stdout to a pipe/file, so `print()` progress rows stay invisible until flush/exit; `-u`
streams them live to the log. Only stderr is unbuffered by default, so without `-u` mid-run progress can't
be monitored.

GPU-torch setup, troubleshooting, and enabling another venv → **`~/docs/torch-gpu.md`** (installer:
`~/scripts/install_rocm_torch.sh`). Don't delete the `rocm/pytorch` base image — it's the torch source.

## Performance gate — before any heavy run

A run must pass the perf gate before launch:

- **Optimal scope** — the smallest run that answers the question. Fast-iterate on one sweep setting.
- **Max CPU/GPU utilization** — uses batching, concurrency, SIMD-optimization where appropriate.

Refine the implementation and run plan with `/auto-review-loop` against the standardized perf prompt
(`scripts/harness/perf_gate.md`) until it passes. Estimate wall-time, and if it exceeds 10 min, confirm saturation first.

## Docs (`docs/**/*.md`)

Frontmatter required: `type` (handoff·plan·prototype-note·research·theory·dev-log·reference),
`status` (current·partial·stale), `created`, `updated`, `tags`; optional `superseded_by`,
`supersedes`, `companion`, `archive_reason`. Folder by `type`: handoff→`docs/handoffs/YYYY-MM-DD-<slug>.md`,
plan|reference→`docs/plans/`, research|theory→`docs/research/`, prototype-note→`docs/dev/prototype/`,
dev-log→`docs/dev/logs/`. Archive inactive handoffs to `docs/archive/handoffs/`; stale plans stay
put with `status: stale` + `archive_reason`. `companion:` references repo-local docs only.
`docs/research/` docs need a Definitions glossary (the IT vocabulary is dense and cross-community).

## Research wiki (`research-wiki/`)

- **Registering a paper (`research-wiki/papers/`): never leave a bare scaffold.** Fill the key
  relevant information — use the template sections as a loose guide, add new sections where they
  fit. Every page must say **why the paper was surfaced** and **why it's relevant to the research
  topic at hand** (the `Relevance to This Project` section), plus its concrete Key Results and the
  design question it bears on (substitution/extraction architecture, privacy attack, RANTEXT-family
  limitation, …). `_TODO._`-only pages are not acceptable.
- **When writing a `docs/research/` report, link the registered papers it draws on** from
  `research-wiki/papers/` (relative path), rather than citing raw URLs alone — the wiki page is the
  canonical entry. If a cited paper isn't yet registered, register it first.
- **Every research-wiki reference in `docs/**/*.md` must also carry the paper's remote identifier as a
  link** — `([arXiv 1234.56789](https://arxiv.org/abs/1234.56789))` (or DOI / ACL / venue link when there
  is no arXiv), placed right after the wiki-page link, inline and in any Sources list. Pull the id from the
  page's `external_ids` frontmatter, never from memory.

## Training experiments — spec-then-results in `research-wiki/training/`

Every fine-tuning / training run gets a record at
`research-wiki/training/YYYY-MM-DD-<TASK>-<component>-v<N>-<slug>.md`, where:
- **date** = run date;
- **`<TASK>`** = training method, UPPERCASE — `FT` (supervised fine-tune), `RL` (reinforcement learning), … ;
- **`<component>`** = the model being trained — `detector`, `ranker`, … ;
- **`v<N>`** = the run's version, **monotonic per `<TASK>-<component>` track**, independent of date
  (FT-detector: v1→v2→v3→v4; RL-ranker: v1→v2). A new run in a track takes the next number; a completed
  run's number never changes;
- **`<slug>`** = the distinguishing method/feature.

Example: `2026-07-05-FT-detector-v4-base-genfirst-mix.md`. Refer to runs by track+version ("FT-detector v3",
"the v4 base+mix run"). **Write the spec _before_ the run, fill the results _after_** — the same doc carries both.
- Frontmatter: `type: training-experiment`, `status` (planned·running·done), `created`, `model` (init
  checkpoint), `dataset` (train-mix summary), `result` (one-line headline or `pending`), `tags`,
  `companion` (the `docs/research/` report).
- Sections: **Objective & hypothesis · Training data (sources + explicit ratios + type-mapping) ·
  Training config · Selection & operating point · Evaluation & success criteria · Results (measured or
  `pending`) · Ablations · Cost · Risks & caveats · Artifacts (paths) · Sources**.
- On completion: set `status: done`, fill **Results** with measured numbers (report the win *and* any
  regression — empirical-honesty rule), and cross-link predecessor/successor runs.
- These live under `research-wiki/` but are **not** in the paper index (`research_wiki.py rebuild_index`
  covers papers/ideas/experiments/claims only). Link the companion `docs/research/` report both ways.

## Scripts — one-time spikes go in `scripts/spikes/`

One-off / throwaway scripts (ad-hoc comparisons, qualitative demos, exploratory probes) live in
`scripts/spikes/`, **not** `scripts/`. `scripts/` is for durable, re-run workflows (gate, sweeps,
corpus builders, training). If a spike graduates into a reusable workflow, move it up to `scripts/`.

## Naming — no plan-level or doc-internal identifiers outside their defining doc

Numbered/lettered doc-internal identifiers — **plan/phase (`D1`, `P0`, `RD4`), requirement/property
(`P1`–`P9`), experiment arms (`Arm A`/`Arm B`), and the like** — live **only in the `docs/plans/` or
`docs/research/` file that defines them**, where the numbering is spelled out. **Never** use them in
code (file names, module/function names, data/result paths), HTML pages, `research-wiki/` records, or
cross-doc references — name after the **method / feature / property it denotes** (`latticecloak_tau_sweep.py`,
`results/latticecloak_detection_gate.json`; "open-label generality", not "P7"; "the knowledgator+TAB
fine-tune", not "Arm B"). Citing a plan/research *document path* in a docstring is fine; baking its
numbering into an identifier is not — the numbers get renumbered, the self-descriptive name does not.

**Exception — training-record versions.** The `v<N>` in a `research-wiki/training/` filename (per the
Training-experiments schema above) IS an allowed, canonical identifier. It denotes the run's monotonic
position in its `<TASK>-<component>` lineage — defined by the training-record series itself, not by a plan —
and is **stable** (a completed run's version never renumbers, unlike plan/phase numbers). So `FT-detector v3`
in a filename, cross-reference, or discussion is correct. This exception is narrow: it does **not** license
plan/phase/requirement/arm identifiers (`P7`, `D1`, `Arm B`) anywhere.
