---
type: benchmark-result
status: done
created: 2026-07-07
updated: 2026-07-07
benchmark: roundtrip-pipeline
suite: primary_utility
result: placeholder baseline is the only arm with nonzero offline privacy, but utility collapses on the clinical item
tags: [roundtrip, benchmark, baseline, privacy-utility, smoke]
---

# Roundtrip Baseline Smoke

## Objective

Run a tiny explicit-model baseline of the roundtrip pipeline:

```text
doc_orig -> detector + substitutor -> doc_p -> gemma 4 (E4B) -> out_p -> extractor -> out_final
```

This is a harness smoke, not a publication-grade result. It uses three documents, one from each release-one
primary utility domain.

## Configuration

- Suite: `primary_utility`
- Limit: `3`
- Seed: `0`
- Items:
  - `clinical/aci/D2N001`
  - `lexsum/lexsum/EE-MO-0105`
  - `wikibio/wikibio/harald-czudaj`
- Detector: `current`
- Detector model: `data/models/pii_gliner_finedem/final`
- Detector fine-dem mode: `true`
- Remote model: `gemma 4 (E4B)`
- Remote endpoint: `OPENAI_BASE_URL=http://localhost:8060/v1`
- Extractor version: `current`
- Extractor model manifest id: `all-MiniLM-L6-v2`
- Attacker tier: `offline-v1`
- Attacker model manifest ids: `offline-docp`, `offline-reconstruct`, `offline-leak`
- LLM cache: `INFERDPT_LLM_CACHE=data/llm_cache`

## Commands

```bash
INFERDPT_LLM_CACHE=data/llm_cache \
OPENAI_BASE_URL=http://localhost:8060/v1 \
PYTHONPATH=src \
.venv/bin/python -u scripts/run_roundtrip_benchmark.py \
  --suite primary_utility \
  --limit 3 \
  --seed 0 \
  --detector-version current \
  --detector-model data/models/pii_gliner_finedem/final \
  --detector-fine-dem \
  --substitutor no_privacy \
  --privacy-setting tau=0.02 \
  --remote-model "gemma 4 (E4B)" \
  --extractor-version current \
  --extractor-model all-MiniLM-L6-v2 \
  --attacker-version offline-v1 \
  --attack-docp-model offline-docp \
  --attack-reconstruction-model offline-reconstruct \
  --attack-leak-model offline-leak \
  --output-dir results/roundtrip_benchmark/baseline-smoke-no_privacy
```

The same command was repeated with `--substitutor all_placeholder` and `--substitutor coarsest_text`.

## Headline Results

| Arm | Config hash | Mean sensitive fact recall | Mean ROUGE-L | ROUGE-L CI | Realized privacy | Unsupported insertions | Detector residuals |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_privacy` | `4835d7b2aa4ede55` | 0.7965 | 0.3125 | [0.2379, 0.4486] | 0.0000 | 24 | 26 |
| `coarsest_text` | `58d0cc5bfb7129f0` | 0.6426 | 0.2762 | [0.2089, 0.3661] | 0.0000 | 19 | 26 |
| `all_placeholder` | `c676f0aa34fdd7dd` | 0.3333 | 0.1930 | [0.0000, 0.3456] | 0.1111 | 9 | 26 |

Interpretation: the current deterministic offline attacker gives `all_placeholder` the only nonzero realized
privacy score, but its utility collapses on the clinical item. `coarsest_text` recovers much of the utility
gap relative to `no_privacy`, but the offline attacker still scores privacy as broken because generalized or
restored surfaces leak enough exact information under this simple metric.

## Per-Domain Results

| Arm | Domain | Item | Sensitive fact recall | ROUGE-L | Echoed spans | Restored echoed spans | Unsupported insertions | Detector residuals |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `no_privacy` | clinical | `clinical/aci/D2N001` | 0.9730 | 0.2638 | 22 | 22 | 21 | 0 |
| `no_privacy` | legal | `lexsum/lexsum/EE-MO-0105` | 0.6667 | 0.4486 | 10 | 9 | 1 | 24 |
| `no_privacy` | biography | `wikibio/wikibio/harald-czudaj` | 0.7500 | 0.2250 | 19 | 18 | 2 | 2 |
| `coarsest_text` | clinical | `clinical/aci/D2N001` | 0.5946 | 0.2089 | 11 | 10 | 14 | 0 |
| `coarsest_text` | legal | `lexsum/lexsum/EE-MO-0105` | 0.5833 | 0.3661 | 8 | 8 | 3 | 24 |
| `coarsest_text` | biography | `wikibio/wikibio/harald-czudaj` | 0.7500 | 0.2535 | 16 | 12 | 2 | 2 |
| `all_placeholder` | clinical | `clinical/aci/D2N001` | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 |
| `all_placeholder` | legal | `lexsum/lexsum/EE-MO-0105` | 0.5000 | 0.3455 | 2 | 2 | 5 | 24 |
| `all_placeholder` | biography | `wikibio/wikibio/harald-czudaj` | 0.5000 | 0.2333 | 7 | 7 | 4 | 2 |

## Artifacts

- `results/roundtrip_benchmark/baseline-smoke-no_privacy/`
- `results/roundtrip_benchmark/baseline-smoke-coarsest_text/`
- `results/roundtrip_benchmark/baseline-smoke-all_placeholder/`

Each artifact directory contains:

- `manifest.json`
- `items.jsonl`
- `traces.jsonl`
- `stage_metrics.json`
- `privacy_metrics.json`
- `utility_metrics.json`
- `matched_privacy_frontier.json`
- `report.md`

## Caveats

- This is `n=3`, one item per primary utility domain. Treat it as a harness smoke and qualitative baseline,
  not a stable frontier.
- The `offline-v1` attacker is exact/canonical and deterministic. It is useful for catching obvious leaks,
  but it is not the LLM re-identification attacker required for publication claims.
- Gold-sensitive spans are still bootstrap spans from the benchmark registry, not final curated gold. The
  legal item has many registry false positives such as title-cased legal phrases, which explains the large
  detector residual count.
- `unsupported_insertion_count` is inflated for `no_privacy` and partially confounded for text baselines
  because the current metric treats original surfaces that appear without an echoed replacement as unsupported
  insertions. This is useful as a safety alarm, but not yet a clean extractor-only error rate.
- The all-placeholder clinical item produced no useful restored task output under this prompt/model slice.

## Decision

Keep `no_privacy`, `coarsest_text`, and `all_placeholder` as required baseline arms. For the next benchmark
iteration, fix the span-gold bootstrap and unsupported-insertion accounting before making any method claim.
