# agent-cloak — privacy layer for closed-box LLM inference

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
