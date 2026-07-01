# agent-cloak — RANTEXT / InferDPT embedding-map (φ) research

## Objective
Study the embedding map **φ** in RANTEXT (InferDPT's per-token ε-LDP perturbation): does the choice of φ
move the outcomes that matter? **The only metrics that matter: privacy + inversion/attacks vs utility, at a
fixed ε.** Geometry/composition proxies (retention, `syn_prec`, `|C_r|` size, anisotropy, eff_dim) are
*diagnostics only* — never goals, never the basis of a model comparison.

## Empirical honesty (hard rule — do not break)
- Compare candidates (φ, mechanisms) **only at fixed ε and identical mechanism settings**. **Never** invent
  or implicitly apply a per-model calibration/normalization knob (e.g. tuning `noise_scale` to equalize
  `|C_r|` or retention): it places models at *different real privacy levels* and invalidates the
  privacy↔utility comparison. This already produced a wrong conclusion once.
- Measure privacy/attack/utility as **outcomes**. If the mechanism degenerates at fixed settings (e.g.
  `|C_r|`→100% saturation / curse of dimensionality), **that is the finding — report it, don't engineer
  around it.**
- Move operating points only with the legitimate knob (**ε**); to compare across operating points use an
  ε-sweep / Pareto curve at equal *realized* privacy — never a per-model fudge factor.
- No result claims without the run output; state degeneracies, confounds, and caveats plainly.

## GPU — run heavy workflows in the host `.venv`

The host `.venv` **runs on the GPU directly** — no container. Use it for **all** heavy workflows
(capture, PVI/MDL/CLUB probes, inversion attacks, `talens.cli`, `calibrate_capture`) and for
`pytest`:

```bash
.venv/bin/python src/inferdpt/eval.py --caches data/vocab_cf,data/vocab_qwen_sub --attacks
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
  RANTEXT-family limitation(s) it bears on. `_TODO._`-only pages are not acceptable.
- **When writing a `docs/research/` report, link the registered papers it draws on** from
  `research-wiki/papers/` (relative path), rather than citing raw URLs alone — the wiki page is the
  canonical entry. If a cited paper isn't yet registered, register it first.
