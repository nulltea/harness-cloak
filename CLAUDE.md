
## GPU — wrap every heavy command

The host `.venv` is **CPU-only torch**; use it for model-free `pytest` only. Anything touching
real Qwen3 activations (capture, PVI/MDL/CLUB probes, inversion attacks, `talens.cli`,
`calibrate_capture`) MUST run in the ROCm container — it silently falls back to CPU otherwise:

```bash
scripts/run_in_rocm.sh python3 -m talens.cli --corpus corpora/dev-24.txt --control all --out results/run.json
# sanity: scripts/run_in_rocm.sh python3 -c 'import torch; print(torch.cuda.is_available())'
```

One AMD Strix Halo iGPU (gfx1151). **One GPU process at a time**: kill stray containers first;
wait on long runs, never poll-spin. Base image rationale: `Containerfile`.

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
