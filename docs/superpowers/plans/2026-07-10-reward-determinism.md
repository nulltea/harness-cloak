# Reward determinism under concurrency — implementation plan

Approved design (2026-07-10, brainstorm in-session; spec updated: `docs/specs/RL/roundtrip-ranker-infiller.md`
§Determinism under concurrency): single-flight generation + cache-bypass refresh + ExIt winner
re-verification. Fixes issue register §1b (`docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md`).

## Why this shape (context for implementers)

llama.cpp batched inference makes temp-0 outputs depend on concurrent batch composition.
`LLMClient.chat` content-addresses every call on `(model, base_url, messages, params)`, so the
defect is **cache-cold computation under concurrency**: whatever batch-dependent value is
computed first gets frozen. Fix at the source: at most one in-flight request to the gen model
(single-flight), so the frozen value is the canonical serial temp-0 output. Reader traffic
targets a different served model process, so it stays concurrent (keeps most of the measured
3.1× speedup). ExIt winners get re-verified with cache-bypassed reader answers because
select-by-max harvests residual jitter into SFT labels — and a cache hit would make
re-verification vacuous.

## Global constraints

- **No GPU / proxy / network calls in any task.** All tests mock the OpenAI client and use tmp
  cache dirs. The probe spike (Task 4) is written but NOT run — its run is user-gated.
- `refresh` must never change the cache key: same file path read/written whether refresh is
  set or not; `refresh` must not leak into request params sent to the server.
- The single-flight lock must NOT be held during cache-hit reads — only around the actual
  network computation (cache miss or refresh).
- Locks are shared per `(base_url, model)` across all `LLMClient` instances (class-level
  registry), so every construction site of a gen-model client serializes against the others.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/<file> -q`. Match existing
  test style in `src/cloak/tests/test_roundtrip.py` / `test_train_roundtrip_mode.py`.
- Minimal diffs (ponytail); no new dependencies; no speculative options.

## Task 1 — `LLMClient`: `refresh` + `single_flight`

File: `src/cloak/llm.py`. Tests: `src/cloak/tests/test_llm_client.py` (new).

- `chat(messages, *, refresh: bool = False, **overrides)`: when `refresh=True`, skip the
  cache read, recompute, and overwrite the cache file. Pop `refresh` before params are built
  so it never enters `_cache_path` input or the request. Thread through `generate` too
  (keyword-only).
- Constructor flag `single_flight: bool = False`. When set, the network call (cache-miss or
  refresh path only) runs under a `threading.Lock` from a class-level registry keyed
  `(base_url, model)` (registry access guarded by a module lock). Two clients with the same
  key share the lock.
- Tests (fake `self._client` via monkeypatch; `CLOAK_LLM_CACHE` = tmp dir):
  1. refresh bypasses an existing cache entry, recomputes, overwrites; subsequent normal call
     reads the new value; cache path identical with/without refresh.
  2. `refresh` never appears in the params the fake transport receives.
  3. single_flight: N threads calling cache-cold `chat` never overlap inside the transport
     (max-concurrency counter == 1); without the flag, overlap occurs (counter > 1).
  4. lock sharing: two separate instances, same (base_url, model), serialize against each
     other; different model → different lock (no serialization).
  5. cache hits do not acquire the lock (e.g., a held lock does not block a cache-hit read —
     acquire the registry lock in the test, then serve a cache hit).

## Task 2 — thread single-flight + reader refresh through the reward path

Files: `src/cloak/train/roundtrip.py`, `src/cloak/train/reward.py`. Tests: extend
`src/cloak/tests/test_roundtrip.py`.

- `roundtrip.py::_remote()`: construct the gen client with `single_flight=True`. Update the
  module docstring: determinism = pinned temp-0 + **single-flight** + cache (cite spec
  §Determinism under concurrency).
- `roundtrip_batch(jobs, workers=6, reader_refresh=False)`: pass `refresh=reader_refresh`
  down to the reader (`fact_f1s`). Gen is NOT refreshed (its cached value is canonical under
  single-flight).
- `reward.py`: `fact_f1s(out_final, probes, refresh=False)` →
  `_read_batch(questions, context, refresh=False)` → `client.generate(..., refresh=refresh)`.
  `_qa_answer` unchanged (default False).
- Audit: grep `RT_MODEL`, `"gemma 4 (E4B)"` across `src/` and `scripts/` for other gen-client
  constructions (anchor building, reward gate, support-scan spike). Add `single_flight=True`
  to each client that targets the gen model. Report the list of touched sites.
- Tests: fake LLM layer records (a) the gen client was constructed with `single_flight=True`,
  (b) `reader_refresh=True` reaches reader `generate` calls as `refresh=True` and the default
  is False, (c) gen calls never receive `refresh=True`.

## Task 3 — ExIt winner re-verification

File: `scripts/train_ranker.py` (`exit_round`). Tests: extend
`src/cloak/tests/test_train_roundtrip_mode.py`.

- In `exit_round`, keep each doc's baseline job and each rollout's job addressable (they are
  already built into `jobs`; retain references per doc).
- After the group pass selects candidates (`best_r > bc_r[di]`), re-verify each candidate
  **serially, after the batch completes**:
  `clean_win = roundtrip_batch([win_job], workers=1, reader_refresh=True)[0]["recall"] or 0.0`
  and the same for the doc's baseline job; keep the winner iff `clean_win > clean_bc`.
- Stats: extend the returned dict with `n_candidates` (pre-verification) and
  `n_verify_dropped` (candidates rejected by the clean comparison). `n_winners` stays the
  post-verification count.
- Tests: stub `roundtrip_batch` so the group pass reports an inflated winner and the
  verification pass reports clean values that (case A) still win → kept, (case B) no longer
  win → dropped and counted in `n_verify_dropped`. Assert verification calls use
  `workers=1, reader_refresh=True` and happen once per candidate + once per candidate-doc
  baseline.

## Task 4 — gen-determinism + reader-jitter probe (write only; run is user-gated)

File: `scripts/spikes/gen_determinism_reader_jitter_probe.py` (new; pattern of
`scripts/spikes/reader_parallelism_smoke.py` — reuse its env/arms/probes loading flags).

Implements spec gate 1 (§Gates):
- **Arm gen-det**: same jobs through `roundtrip_batch` at `workers=6` (single-flight gen +
  parallel readers) vs `workers=1`, each arm under its own fresh scratch cache dir
  (`CLOAK_LLM_CACHE` per arm, under the run's scratch root) so gen recomputes; report
  per-job `out_p` exact-match rate (expect 100% — validates the single-flight assumption).
- **Arm reader-jitter**: fixed `out_final` texts (from the workers=1 arm), reader answers at
  `workers=1` vs `workers=6` with `refresh=True` (cache bypass); report per-answer flip rate
  and per-doc recall delta distribution.
- Output: JSON under `results/` + printed summary rows. `-u`-friendly (flush prints).
- No test file (spike); a `--help` smoke via `py_compile`/argparse suffices. DO NOT run the
  probe: GPU + proxy + shared box (one GPU process; occupancy check + user confirmation).

## Out of scope (tracked elsewhere)

- Running the probe, fresh-cache support-scan re-run, pilot: user-gated GPU work after merge.
- Blocker 2 (per-level counts → legality mask) and blocker 3 (ladder reward wiring): separate
  loops.
