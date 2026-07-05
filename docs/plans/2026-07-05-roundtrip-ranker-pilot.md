# Round-Trip Ranker Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [rl, round-trip-reward, ranker, pilot, expert-iteration, rloo, probes, support-scan]
companion: [docs/specs/RL/roundtrip-ranker-infiller.md,
            docs/plans/2026-07-05-roundtrip-rl-strategy.md]
---

**Goal:** Implement everything the Stage-1 (ranker) round-trip RL pilot needs — round-trip
reward, validated probes, support-scan gate, ExIt + RLOO trainer modes, encoder policy,
saturation probe — per `docs/specs/RL/roundtrip-ranker-infiller.md`. π_fill (infiller) is
deferred to Stage 2 and appears in NO task.

**Architecture:** The round-trip reward is a thin composition of machinery that already exists
(`LLMClient` disk cache + `pmap` + `invert` + `fact_recall` — see `scripts/reward_gate.py:103-113`
for the exact pattern). New code = one reward module, one probe-validation script, one support
scan, trainer modes, one policy class, one perf spike. Heavy *runs* are NOT tasks — they live in
the gated runbook at the end.

**Tech Stack:** Python 3.11+, torch (ROCm host `.venv`), transformers, existing `cloak`/`inferdpt`
packages, llama-swap proxy at the default base URL.

## Global Constraints

Copied verbatim from the spec / project rules — every task's requirements include these:

- **Pinned reward model:** `RT_MODEL = "LFM2.5-8B-A1B"`, temperature 0.0, max_tokens 512, cache
  key = content hash of (model, messages, params) via `INFERDPT_LLM_CACHE`. Changing any of these
  re-gates.
- **Reward = graded mean token-F1** (the deployed `fact_recall` definition). The validation
  threshold `TH = 0.5` binarizes only probe keep/drop, never the reward.
- **Probe teacher = "gemma 4 (E4B)"** (existing `cloak.train.probes` machinery) — must never be
  the reward model.
- **Docs with < 3 surviving train probes are excluded from the RL reward** and listed in the
  probe-health report — never silently kept.
- **Privacy is floors-only** — no reward term may read `p6`/`walk_risk`/aset as a reward signal
  in round-trip mode; legality masks are unchanged.
- **No cross-floor averaging**; greedy read-outs at fixed env floors only. **G ≥ 2** enforced.
- **Detection is never re-run inside tasks** — all consumers load the frozen arms artifact.
- Tests must run **offline** (monkeypatch the LLM client / round-trip function). Anything that
  hits the proxy or downloads a model lives behind `__main__` live-smokes or the runbook.
- **Do not touch:** `research-wiki/training/*detector*`, `scripts/build_pii_span_dataset.py`,
  `.gitignore`, `data/models/` (user's detector work in flight). Do not modify or delete
  `data/task_arms_tau0.02.json`, `data/ranker_env.json`, `data/llm_cache/`, `data/surrogate_probes.json`.
- Long/background jobs run **unbuffered** (`.venv/bin/python -u`). GPU: llama-swap is the
  resident GPU server; local torch smokes fall back to CPU if `pgrep -af train_pii` shows the
  user's detector run live.
- One-off scripts → `scripts/spikes/`; durable workflows → `scripts/`. No plan-internal
  identifiers in code/file names (name by method/feature).
- Run all tests as: `PYTHONPATH=src:scripts .venv/bin/python -m pytest <file> -v`
  (repo tests live in `src/cloak/tests/`).
- Commit per task; **never push**.

## Existing interfaces (verbatim — do not re-derive)

- `LLMClient(model, temperature=0.0, max_tokens=N, extra_body={"chat_template_kwargs": {"enable_thinking": False}})`
  → `.generate(prompt) -> str`; disk-cached when `INFERDPT_LLM_CACHE` is set
  (`src/inferdpt/llm.py`).
- `pmap(fn, jobs, workers=N) -> list` (`src/inferdpt/pipeline.py`).
- `invert(text, R) -> (inverted_text, stats)` (`src/cloak/extract.py`).
- `fact_recall(out_final, probes) -> float | None`; probes = `[{"surface": str, "question": str}]`
  (`src/cloak/train/reward.py`; per-probe loop: reader `_qa_answer(p["question"], out_final)`,
  then `token_f1(answer, p["surface"])`, mean).
- `TASK_TEMPLATE[corpus].format(doc=doc_p)` (`src/cloak/tasks.py`).
- `probes_for_docs(docs, R_of, workers=6) -> {doc_id: [{"surface", "question"}]}` — gemma
  teacher, cached at `data/surrogate_probes.json` (`src/cloak/train/probes.py`).
- `load_task_docs(corpus, n) -> [{"id", "text", ...}]`, `refs_of(doc) -> (gold, ...)`
  (`src/cloak/corpora.py`).
- Env: `data/ranker_env.json` = `{"k_floors": {...}, "corpora": {corpus: {doc_id: {"trainable",
  "spans", "probes": {"train", "heldout"}}}}}`. Arms artifact via
  `load_artifact()` (`scripts/build_arms_artifact.py`): `art[corpus][doc_id]` has
  `"tau_walk"/"all_floor"/"suppression"` = `(doc_p, R)` tuples.
- Trainer (`scripts/train_ranker.py`): `assemble(text, R_walk, spans, choice) -> (doc_p, R)`;
  `derive_spans(raw_spans, floors, corpus, device) -> (spans, feats)` (adds `legal`,
  `bc_action`); `rollout_reward(doc, span_rows, feats, policy, alpha, greedy=False)
  -> (r, parts, logps)`; `behavior_clone(...)`; `sample_floors(floors, rng)`; docs rows carry
  `{"id","corpus","text","R_walk","raw_spans","spans","feats","probes_train"}`.
- Policy (`src/cloak/train/ranker.py`): `RankerPolicy` with `.sample(feats, legal, greedy)
  -> (action_idx, logp)` and `.log_probs(feats, legal) -> tensor[len(legal)]`;
  `action_features(span, corpus, k) -> tensor[n_actions, N_FEAT]`, `N_FEAT = 17`.
- The all-placeholder choice pattern (from `scripts/reward_gate.py:94`):
  `ph_choice = {s["surface"].lower(): s["actions"][-1] for s in spans}` (last action is always
  the placeholder).

---

### Task 1: Round-trip reward module

**Files:**
- Create: `src/cloak/train/roundtrip.py`
- Modify: `src/cloak/train/reward.py` (extract `fact_f1s` from `fact_recall`)
- Test: `src/cloak/tests/test_roundtrip.py`

**Interfaces:**
- Consumes: `LLMClient`, `pmap`, `invert`, `TASK_TEMPLATE`, `token_f1`, `_qa_answer`.
- Produces (later tasks rely on these exact signatures):
  - `roundtrip.RT_MODEL: str = "LFM2.5-8B-A1B"`, `roundtrip.MAX_TOKENS: int = 512`
  - `roundtrip.roundtrip_batch(jobs: list[dict], workers: int = 8) -> list[dict]` where each
    job is `{"corpus": str, "doc_p": str, "R": list[dict], "probes": list[dict]}` and each
    result is `{"out_p": str, "out_final": str, "f1s": list[float], "recall": float | None}`
  - `reward.fact_f1s(out_final: str, probes: list[dict]) -> list[float]`

- [ ] **Step 1: Extract `fact_f1s` in `src/cloak/train/reward.py`**

Read the existing `fact_recall` body (it loops probes → `_qa_answer(p["question"], out_final)`
→ `token_f1(answer, p["surface"])` → mean). Refactor to:

```python
def fact_f1s(out_final: str, probes: list[dict]) -> list[float]:
    """Per-probe realized token-F1 on out_final (original space — no generalization,
    no inversion). The per-probe form of fact_recall; shared by the round-trip reward,
    probe validation, and the support scan."""
    return [token_f1(_qa_answer(p["question"], out_final), p["surface"]) for p in probes]


def fact_recall(out_final: str, probes: list[dict]) -> float | None:
    # (keep the existing docstring verbatim)
    f1s = fact_f1s(out_final, probes)
    return (sum(f1s) / len(f1s)) if f1s else None
```

Preserve `fact_recall`'s existing docstring and semantics exactly (`None` when no probes).
If the current body has any extra detail (e.g. rounding, per-probe details), keep behavior
identical — `fact_f1s` must return exactly the per-probe values `fact_recall` was averaging.

- [ ] **Step 2: Write the failing test**

```python
# src/cloak/tests/test_roundtrip.py
"""Round-trip reward wiring — offline (LLM + reader monkeypatched)."""
import cloak.train.roundtrip as rt


class _StubClient:
    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


def test_roundtrip_batch_inverts_and_scores(monkeypatch):
    # remote echoes the fill; invert() must map it back; probes scored on out_final
    stub = _StubClient(["Patient is a fifty-something female with chest pain."])
    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "fact_f1s", lambda out, probes: [1.0 if "50" in out else 0.0])
    jobs = [{"corpus": "clinical",
             "doc_p": "a fifty-something female reports chest pain",
             "R": [{"surface": "50-year-old", "type": "DEM", "action": "generalize",
                    "replacement": "fifty-something"}],
             "probes": [{"surface": "50-year-old", "question": "How old is the patient?"}]}]
    res = rt.roundtrip_batch(jobs, workers=1)
    assert len(res) == 1
    assert "50-year-old" in res[0]["out_final"]          # inversion fired
    assert res[0]["recall"] == 1.0 and res[0]["f1s"] == [1.0]
    assert "fifty-something female reports" in stub.prompts[0]   # doc_p reached the template


def test_roundtrip_batch_no_probes_gives_none(monkeypatch):
    stub = _StubClient(["anything"])
    monkeypatch.setattr(rt, "_remote", lambda: stub)
    res = rt.roundtrip_batch([{"corpus": "enron", "doc_p": "x", "R": [], "probes": []}],
                             workers=1)
    assert res[0]["recall"] is None and res[0]["f1s"] == []


def test_fact_f1s_matches_fact_recall(monkeypatch):
    import cloak.train.reward as rw
    monkeypatch.setattr(rw, "_qa_answer", lambda q, c: "42 mg")
    probes = [{"surface": "42 mg", "question": "What dose?"},
              {"surface": "Oslo", "question": "Where?"}]
    f1s = rw.fact_f1s("text", probes)
    assert f1s[0] == 1.0 and f1s[1] == 0.0
    assert rw.fact_recall("text", probes) == sum(f1s) / 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_roundtrip.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: cloak.train.roundtrip` (and `fact_f1s`
missing).

- [ ] **Step 4: Write `src/cloak/train/roundtrip.py`**

```python
"""Round-trip reward (spec docs/specs/RL/roundtrip-ranker-infiller.md, Phase 1).

R_rt = realized fact recall (graded mean token-F1) on out_final over a doc's train-split
probes, where out_final = invert(Remote(task_prompt(doc_p)), R). Deterministic given doc_p:
pinned model, temperature 0, content-addressed disk cache (INFERDPT_LLM_CACHE) — the
determinism is load-bearing (cache = reward memoization = ExIt pool; spec "one subtlety").
"""
import os

from cloak.extract import invert
from cloak.tasks import TASK_TEMPLATE
from cloak.train.reward import fact_f1s

RT_MODEL = "LFM2.5-8B-A1B"   # THE pin (spec components table); changing it re-gates
MAX_TOKENS = 512

_client = None


def _remote():
    global _client
    if _client is None:
        from inferdpt.llm import LLMClient
        assert os.getenv("INFERDPT_LLM_CACHE"), \
            "round-trip reward requires INFERDPT_LLM_CACHE (determinism + cost)"
        _client = LLMClient(RT_MODEL, temperature=0.0, max_tokens=MAX_TOKENS,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return _client


def roundtrip_batch(jobs: list[dict], workers: int = 8) -> list[dict]:
    """jobs: [{corpus, doc_p, R, probes}] -> [{out_p, out_final, f1s, recall}].
    recall = graded mean token-F1 (the deployed fact_recall), None when a job has no probes."""
    from inferdpt.pipeline import pmap
    remote = _remote()
    outs = pmap(lambda j: remote.generate(
        TASK_TEMPLATE[j["corpus"]].format(doc=j["doc_p"])), jobs, workers=workers)
    res = []
    for j, op in zip(jobs, outs):
        out_final, _ = invert(op, j["R"])
        f1s = fact_f1s(out_final, j["probes"])
        res.append({"out_p": op, "out_final": out_final, "f1s": f1s,
                    "recall": (sum(f1s) / len(f1s)) if f1s else None})
    return res


if __name__ == "__main__":
    # LIVE smoke (hits the proxy once; requires INFERDPT_LLM_CACHE and the proxy up):
    #   INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -m cloak.train.roundtrip
    r = roundtrip_batch([{"corpus": "enron",
                          "doc_p": "Please send the Q3 numbers to <PERSON_1> by Friday.",
                          "R": [{"surface": "Alice Kim", "type": "PERSON",
                                 "action": "placeholder", "replacement": "<PERSON_1>"}],
                          "probes": [{"surface": "Alice Kim",
                                      "question": "Who should receive the numbers?"}]}],
                        workers=1)
    print(r[0]["out_p"][:120].replace("\n", " "))
    print("recall:", r[0]["recall"])
    assert r[0]["out_p"].strip(), "empty remote reply"
    print("roundtrip live smoke OK")
```

Note: the pmap lambda must touch only `j["corpus"]`/`j["doc_p"]` (no closure over mutable
state) — results order must match job order (pmap guarantees this; do not add reordering).

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_roundtrip.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the pre-existing reward self-check to prove `fact_recall` unchanged**

Run: `PYTHONPATH=src .venv/bin/python -c "import cloak.train.reward as r; print(r.fact_recall('x', []) is None)"`
Expected: `True`. Also run any existing tests touching reward:
`PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests -k "reward or recall" -v`
Expected: no new failures (record the before/after pass counts in the report).

- [ ] **Step 7: Commit**

```bash
git add src/cloak/train/roundtrip.py src/cloak/train/reward.py src/cloak/tests/test_roundtrip.py
git commit -m "feat: round-trip reward module (pinned LFM2.5, cached) + fact_f1s extraction"
```

---

### Task 2: Probe validation pipeline (anchors + health report)

**Files:**
- Create: `scripts/build_probes.py`
- Test: `src/cloak/tests/test_probe_validation.py`

**Interfaces:**
- Consumes: `roundtrip.roundtrip_batch`, `reward.fact_f1s`, `probes_for_docs`,
  `load_task_docs`, `refs_of`, env/artifact loaders, `assemble` + the `ph_choice` pattern.
- Produces:
  - Artifact `data/probes_validated.json`:
    `{doc_id: {"train": [probe], "heldout": [probe], "rejected": {"ceiling": [probe],
    "floor": [probe]}}}` (probe = `{"surface", "question"}`)
  - Report `results/probe_health.json`: per corpus `{"docs": int, "kept_mean": float,
    "kept_min": int, "ceiling_reject_rate": float, "floor_reject_rate": float,
    "excluded_docs": [doc_id]}` (excluded = < 3 surviving train probes)
  - Pure function `validate_probes(cands, hi_f1s, lo_f1s, th=0.5) -> (kept, rej_ceiling,
    rej_floor)` importable as `from build_probes import validate_probes`

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_probe_validation.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_probes import validate_probes  # noqa: E402


def test_validate_probes_keep_and_reject():
    cands = [{"surface": "metformin 500mg", "question": "What dose?"},   # keep
             {"surface": "March 3", "question": "When?"},                # ceiling reject
             {"surface": "chest pain", "question": "What symptom?"}]     # floor reject
    hi = [1.0, 0.2, 1.0]   # f1 vs ceiling anchor out_final(doc_orig)
    lo = [0.0, 0.0, 0.9]   # f1 vs floor anchor out_final(all_placeholder)
    kept, rej_c, rej_f = validate_probes(cands, hi, lo, th=0.5)
    assert [p["surface"] for p in kept] == ["metformin 500mg"]
    assert [p["surface"] for p in rej_c] == ["March 3"]
    assert [p["surface"] for p in rej_f] == ["chest pain"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_probe_validation.py -v`
Expected: FAIL with `ImportError` (no `build_probes`).

- [ ] **Step 3: Write `scripts/build_probes.py`**

```python
"""Validated probe build (spec Phase 0 step 4): teacher questions + anchor validation.

Per doc: candidate probes from the gemma teacher (cloak.train.probes, cached) -> two anchor
round trips through the PINNED reward model (ceiling = doc_orig, floor = all-placeholder,
both full round trips incl. inversion) -> keep iff ceiling f1 >= TH and floor f1 < TH.
The floor check drops probes the all-placeholder baseline already answers (echoed
placeholders invert perfectly — such probes have no dynamic range above the safest action).

Writes data/probes_validated.json + results/probe_health.json. Docs with < 3 surviving
train probes are listed in excluded_docs (spec: excluded from the RL reward, never
silently kept).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/build_probes.py [--corpora clinical,enron,aeslc]
       [--n-docs 16] [--workers 8] [--th 0.5] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

TH = 0.5
OUT = Path("data/probes_validated.json")
REPORT = Path("results/probe_health.json")


def validate_probes(cands, hi_f1s, lo_f1s, th=TH):
    """Pure keep/drop: probe survives iff answerable at the ceiling anchor AND not already
    answered at the floor anchor. Returns (kept, rejected_ceiling, rejected_floor)."""
    kept, rej_c, rej_f = [], [], []
    for p, hi, lo in zip(cands, hi_f1s, lo_f1s):
        if hi < th:
            rej_c.append(p)
        elif lo >= th:
            rej_f.append(p)
        else:
            kept.append(p)
    return kept, rej_c, rej_f


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs, refs_of
    from cloak.train.probes import probes_for_docs
    from cloak.train.reward import fact_f1s
    from cloak.train.roundtrip import roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="clinical,enron,aeslc")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--th", type=float, default=TH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    report = {"th": args.th, "corpora": {}}

    for corpus in args.corpora.split(","):
        docs = load_task_docs(corpus, args.n_docs)
        per_doc = env["corpora"].get(corpus, {})
        rows = [d for d in docs if d["id"] in per_doc and per_doc[d["id"]]["spans"]]
        # 1. candidate probes (teacher, cached; R = artifact tau_walk R)
        R_of = {d["id"]: art[corpus][d["id"]]["tau_walk"][1] for d in rows}
        cands = probes_for_docs(rows, R_of, workers=args.workers)
        # 2. anchor round trips: ceiling (doc_orig, R=[]) + floor (all-placeholder)
        jobs, meta = [], []
        for d in rows:
            spans = per_doc[d["id"]]["spans"]
            ph_choice = {s["surface"].lower(): s["actions"][-1] for s in spans}
            lo_doc, lo_R = assemble(d["text"], art[corpus][d["id"]]["tau_walk"][1],
                                    spans, ph_choice)
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=args.workers)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = r["out_final"]
        # 3. validate + split
        stats = {"docs": 0, "kept": [], "rej_c": 0, "rej_f": 0, "cand": 0,
                 "excluded_docs": []}
        for d in rows:
            ps = cands.get(d["id"], [])
            if not ps or d["id"] not in anchor:
                continue
            hi = fact_f1s(anchor[d["id"]]["hi"], ps)
            lo = fact_f1s(anchor[d["id"]]["lo"], ps)
            kept, rc, rf = validate_probes(ps, hi, lo, args.th)
            rng = random.Random(args.seed)
            rng.shuffle(kept)
            n_hold = max(1, len(kept) // 4) if len(kept) >= 2 else 0
            out[d["id"]] = {"train": kept[n_hold:], "heldout": kept[:n_hold],
                            "rejected": {"ceiling": rc, "floor": rf}}
            stats["docs"] += 1
            stats["cand"] += len(ps)
            stats["kept"].append(len(kept))
            stats["rej_c"] += len(rc)
            stats["rej_f"] += len(rf)
            if len(out[d["id"]]["train"]) < 3:
                stats["excluded_docs"].append(d["id"])
        n = max(stats["docs"], 1)
        report["corpora"][corpus] = {
            "docs": stats["docs"],
            "kept_mean": round(sum(stats["kept"]) / n, 2),
            "kept_min": min(stats["kept"], default=0),
            "ceiling_reject_rate": round(stats["rej_c"] / max(stats["cand"], 1), 3),
            "floor_reject_rate": round(stats["rej_f"] / max(stats["cand"], 1), 3),
            "excluded_docs": stats["excluded_docs"]}
        print(f"[{corpus}] {report['corpora'][corpus]}", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT} + {REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_probe_validation.py -v`
Expected: 1 passed. (The `main()` body is exercised by the runbook, not tests — it needs the
proxy. Do NOT add tests that import heavy models.)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_probes.py src/cloak/tests/test_probe_validation.py
git commit -m "feat: probe validation pipeline (ceiling/floor anchors) + probe-health report"
```

---

### Task 3: Round-trip support scan (THE training gate)

**Files:**
- Create: `scripts/spikes/roundtrip_support_scan.py`
- Test: `src/cloak/tests/test_support_scan.py`

**Interfaces:**
- Consumes: `roundtrip_batch`, `derive_spans`, `assemble`, env/artifact/probe loaders.
- Produces: `results/roundtrip_support_scan.json` with
  `{"n_swaps", "n_up", "n_down", "max_abs_delta", "mean_probes_per_doc", "verdict",
  "rows": [{"doc_id", "surface", "from", "to", "delta", "probe_flips_up", "probe_flips_down"}]}`;
  pure function `scan_verdict(rows, mean_probes) -> dict` importable from the spike.

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_support_scan.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "spikes"))
from roundtrip_support_scan import scan_verdict  # noqa: E402


def _row(delta, up=0, down=0):
    return {"doc_id": "d", "surface": "s", "from": 0, "to": 1, "delta": delta,
            "probe_flips_up": up, "probe_flips_down": down}


def test_pass_needs_both_directions_and_magnitude():
    rows = [_row(0.15, up=1), _row(-0.2, down=1), _row(0.0)]
    v = scan_verdict(rows, mean_probes=10.0)
    assert v["verdict"] == "PASS" and v["n_up"] == 1 and v["n_down"] == 1


def test_fail_one_direction_only():
    rows = [_row(-0.2, down=1), _row(-0.1, down=1)]
    assert scan_verdict(rows, mean_probes=10.0)["verdict"] == "FAIL"


def test_fail_below_quantization():
    rows = [_row(0.01, up=0), _row(-0.01, down=0)]   # |delta| < 1/mean_probes = 0.1
    assert scan_verdict(rows, mean_probes=10.0)["verdict"] == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_support_scan.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `scripts/spikes/roundtrip_support_scan.py`**

```python
"""Round-trip support scan — THE gate before any RL training run (spec Gates-1; the
round-trip descendant of probe_flip_scan.py, mandated by the 2026-07-05 pivot handoff).

From the floor-walk baseline: single-action counterfactuals (each decision span, each legal
alternative action, capped) -> full cached round trips -> per-probe realized-F1 deltas.
PASS = reward responds in BOTH directions with magnitude above the quantization step.
A support desert is a FINDING about the environment — report it, never work around it.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/roundtrip_support_scan.py \
       [--max-swaps 150] [--workers 8] [--probes data/probes_validated.json] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

OUT = Path("results/roundtrip_support_scan.json")
FLIP = 0.5     # per-probe |delta f1| counting as a flip


def scan_verdict(rows, mean_probes: float) -> dict:
    """PASS iff swaps moved realized recall in both directions AND the largest move
    exceeds the quantization step (1/mean probes per doc)."""
    step = 1.0 / max(mean_probes, 1.0)
    n_up = sum(1 for r in rows if r["delta"] > 0)
    n_down = sum(1 for r in rows if r["delta"] < 0)
    max_abs = max((abs(r["delta"]) for r in rows), default=0.0)
    ok = n_up >= 1 and n_down >= 1 and max_abs >= step
    return {"n_swaps": len(rows), "n_up": n_up, "n_down": n_down,
            "max_abs_delta": round(max_abs, 4), "quant_step": round(step, 4),
            "verdict": "PASS" if ok else "FAIL"}


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble, derive_spans

    from cloak.corpora import load_task_docs
    from cloak.train.roundtrip import roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-swaps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probes", default="data/probes_validated.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    probes_all = json.loads(Path(args.probes).read_text())
    floors = dict(env["k_floors"])
    rng = random.Random(args.seed)

    # baseline floor-walk rollout per doc (dynamic collision rule: first-come keeps the
    # fill, later colliding spans fall back to placeholder — the trainer's walk-order rule)
    docs, base_jobs = [], []
    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
        for doc_id, d in per_doc.items():
            probes = probes_all.get(doc_id, {}).get("train", [])
            if not d.get("trainable") or not d["spans"] or len(probes) < 3:
                continue
            spans, _ = derive_spans(d["spans"], floors, corpus, "cpu")
            used, choice = set(), {}
            for s in spans:
                a = s["actions"][s["bc_action"]]
                if a["mode"] == "level" and a["fill"].lower() in used:
                    a = s["actions"][next(i for i, x in enumerate(s["actions"])
                                          if x["mode"] == "placeholder")]
                if a["mode"] == "level":
                    used.add(a["fill"].lower())
                choice[s["surface"].lower()] = a
            doc_p, R = assemble(texts[doc_id], art[corpus][doc_id]["tau_walk"][1],
                                d["spans"], choice)
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "spans": spans, "raw_spans": d["spans"], "choice": choice,
                         "probes": probes})
            base_jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": probes})
    base = roundtrip_batch(base_jobs, workers=args.workers)
    base_f1s = {d["id"]: b["f1s"] for d, b in zip(docs, base)}
    base_recall = {d["id"]: b["recall"] for d, b in zip(docs, base)}

    # counterfactual swaps: every (span, legal alternative), sampled down to the cap
    swaps = []
    for d in docs:
        for s in d["spans"]:
            cur = d["choice"][s["surface"].lower()]
            for i in s["legal"]:
                a = s["actions"][i]
                if a is cur or (a["mode"] == cur["mode"] and
                                a.get("fill") == cur.get("fill")):
                    continue
                swaps.append((d, s, i))
    rng.shuffle(swaps)
    swaps = swaps[:args.max_swaps]

    jobs = []
    for d, s, i in swaps:
        choice = dict(d["choice"])
        choice[s["surface"].lower()] = s["actions"][i]
        try:
            doc_p, R = assemble(d["text"], d["R_walk"], d["raw_spans"], choice)
        except AssertionError:      # injectivity collision -> unreachable action, skip
            jobs.append(None)
            continue
        jobs.append({"corpus": d["corpus"], "doc_p": doc_p, "R": R, "probes": d["probes"]})
    live = [j for j in jobs if j]
    outs = iter(roundtrip_batch(live, workers=args.workers))

    rows = []
    for (d, s, i), j in zip(swaps, jobs):
        if j is None:
            continue
        o = next(outs)
        deltas = [a - b for a, b in zip(o["f1s"], base_f1s[d["id"]])]
        rows.append({"doc_id": d["id"], "surface": s["surface"],
                     "from": s["bc_action"], "to": i,
                     "delta": round((o["recall"] or 0) - (base_recall[d["id"]] or 0), 4),
                     "probe_flips_up": sum(x >= FLIP for x in deltas),
                     "probe_flips_down": sum(x <= -FLIP for x in deltas)})

    mean_probes = sum(len(d["probes"]) for d in docs) / max(len(docs), 1)
    v = scan_verdict(rows, mean_probes)
    v.update(mean_probes_per_doc=round(mean_probes, 2), rows=rows)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(v, indent=1))
    print({k: v[k] for k in ("n_swaps", "n_up", "n_down", "max_abs_delta", "verdict")})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_support_scan.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/roundtrip_support_scan.py src/cloak/tests/test_support_scan.py
git commit -m "feat: round-trip support scan gate (counterfactual flips, PASS/FAIL verdict)"
```

---

### Task 4: Trainer — round-trip reward mode, RLOO, tie filter, entropy bonus

**Files:**
- Modify: `scripts/train_ranker.py`
- Test: `src/cloak/tests/test_train_roundtrip_mode.py`

**Interfaces:**
- Consumes: `roundtrip_batch` (Task 1), `data/probes_validated.json` (Task 2 format).
- Produces (Task 5/6 build on these exact names):
  - `sample_rollout(doc, span_rows, feats, policy, greedy=False) -> (choice, logps, ph_rate,
    doc_p, R)` — the sampling half of the old `rollout_reward` (which stays, for surrogate mode,
    reimplemented on top of `sample_rollout`).
  - CLI: `--reward {surrogate,roundtrip}` (default `surrogate`), `--probes PATH` (default
    `data/probes_validated.json`, used only in roundtrip mode), `--adv {group,rloo}`
    (default: `group` for surrogate, `rloo` for roundtrip), `--entropy-coef FLOAT`
    (default 0.0 surrogate / 0.01 roundtrip), `--rt-workers INT` (default 8).
  - In roundtrip mode: kl_coef defaults to 0.0 (explicit `--kl-coef` still wins), alphas are
    ignored (single run, tag `rt`), docs with < 3 validated train probes are dropped at load
    (count printed), and each epoch logs `ties_skipped`.

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_train_roundtrip_mode.py
"""Round-trip trainer mode — offline: roundtrip_batch monkeypatched with a deterministic
fake that rewards keeping level fills (so RLOO has a real gradient direction)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402


def _doc():
    actions = [{"mode": "level", "fill": "a biguanide", "aset": 500.0, "p6": 0.8},
               {"mode": "placeholder", "fill": "<QUANTITY_1>"}]
    span = {"surface": "metformin", "type": "QUANTITY", "start": 0, "actions": actions}
    raw = [dict(span)]
    spans, feats = tr.derive_spans(raw, {"QUANTITY": 100.0}, "clinical", "cpu")
    return {"id": "d0", "corpus": "clinical", "text": "metformin daily",
            "R_walk": [{"surface": "metformin", "type": "QUANTITY", "action": "generalize",
                        "replacement": "a biguanide", "start": 0, "end": 9,
                        "lattice": ["a biguanide"]}],
            "raw_spans": raw, "spans": spans, "feats": feats,
            "probes_train": [{"surface": "metformin", "question": "What drug?"}]}


def fake_roundtrip(jobs, workers=1):
    # reward 1.0 iff the level fill survived into doc_p, else 0.0
    return [{"out_p": "", "out_final": j["doc_p"], "f1s": [float("biguanide" in j["doc_p"])],
             "recall": float("biguanide" in j["doc_p"])} for j in jobs]


def test_sample_rollout_shapes():
    doc = _doc()
    policy = tr.RankerPolicy()
    choice, logps, ph_rate, doc_p, R = tr.sample_rollout(doc, doc["spans"], doc["feats"],
                                                         policy)
    assert set(choice) == {"metformin"} and len(logps) == 1
    assert isinstance(doc_p, str) and isinstance(R, list)


def test_rloo_advantage_no_std():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = tr.rloo_advantage(r)
    # b_g = mean of others: adv_0 = 1 - 0 = 1.0; adv_j = 0 - 1/3
    assert torch.allclose(adv, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))


def test_roundtrip_epoch_moves_policy(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    torch.manual_seed(0)
    policy = tr.RankerPolicy()
    before = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach().clone()
    stats = tr.train_roundtrip([doc], policy, G=4, epochs=3, lr=0.05,
                               entropy_coef=0.01, kl_coef=0.0, ref=None,
                               rt_workers=1, seed=0)
    after = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach()
    assert not torch.allclose(before, after)          # first-smoke movement canary
    assert after[0] > before[0]                       # level action (rewarded) went UP
    assert "ties_skipped" in stats[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_train_roundtrip_mode.py -v`
Expected: FAIL with `AttributeError` (`sample_rollout`, `rloo_advantage`, `train_roundtrip`
missing).

- [ ] **Step 3: Implement in `scripts/train_ranker.py`**

3a. Split sampling from scoring (keep `rollout_reward` working for surrogate mode):

```python
def sample_rollout(doc, span_rows, feats, policy, greedy=False):
    """Sampling half of a rollout under the DYNAMIC injectivity mask (spec §3.3-1).
    Returns (choice, logps, ph_rate, doc_p, R) — no reward computed here."""
    used: set[str] = set()
    choice, logps, n_level = {}, [], 0
    for s, f in zip(span_rows, feats):
        legal_dyn = [i for i in s["legal"]
                     if s["actions"][i]["mode"] == "placeholder"
                     or s["actions"][i]["fill"].lower() not in used]
        a_idx, lp = policy.sample(f, legal_dyn, greedy=greedy)
        a = s["actions"][a_idx]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
            n_level += 1
        choice[s["surface"].lower()] = a
        logps.append(lp)
    doc_p, R = assemble(doc["text"], doc["R_walk"], span_rows, choice)
    return choice, logps, 1.0 - n_level / len(span_rows), doc_p, R
```

Rewrite the body of `rollout_reward` to call `sample_rollout` and then compute A/u_qa/r
exactly as before (byte-identical surrogate behavior — the smoke in Step 5 proves it).

3b. RLOO advantage + entropy helper:

```python
def rloo_advantage(rt: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline, NO std normalization (Dr.GRPO correction; spec Phase 2)."""
    G = rt.numel()
    return (rt - rt.mean()) * G / (G - 1)


def policy_entropy(policy, feats, legal) -> torch.Tensor:
    lp = policy.log_probs(feats, legal)
    return -(lp.exp() * lp).sum()
```

3c. The round-trip training loop (used by main() and by Task 4's test directly):

```python
def train_roundtrip(docs, policy, *, G, epochs, lr, entropy_coef, kl_coef, ref,
                    rt_workers, seed, log_rows=None):
    """RLOO + tie-filter epoch loop against roundtrip_batch. Returns per-epoch stat rows."""
    from cloak.train.roundtrip import roundtrip_batch as _rt   # patchable module attr
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rows = []
    for epoch in range(epochs):
        rng = random.Random(seed * 1000 + epoch)
        order = list(range(len(docs)))
        rng.shuffle(order)
        ep = {"r": [], "ph": [], "ent": [], "ties_skipped": 0}
        for di in order:
            doc = docs[di]
            outs, logps_l, ph_l = [], [], []
            jobs = []
            for _ in range(G):
                choice, logps, ph, doc_p, R = sample_rollout(doc, doc["spans"],
                                                             doc["feats"], policy)
                jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                             "probes": doc["probes_train"]})
                logps_l.append(logps)
                ph_l.append(ph)
            res = roundtrip_batch(jobs, workers=rt_workers)
            rt = torch.tensor([r["recall"] or 0.0 for r in res])
            ep["r"].append(rt.mean().item())
            ep["ph"].append(sum(ph_l) / G)
            if rt.max() == rt.min():                      # DAPO tie filter
                ep["ties_skipped"] += 1
                continue
            adv = rloo_advantage(rt)
            pg = -sum(a * torch.stack(lp).sum() for a, lp in zip(adv, logps_l)) / G
            ent = sum(policy_entropy(policy, f, s["legal"])
                      for s, f in zip(doc["spans"], doc["feats"])) / len(doc["spans"])
            loss = pg - entropy_coef * ent
            if kl_coef > 0 and ref is not None:
                loss = loss + kl_coef * sum(
                    kl_to_ref(policy, ref, f, s["legal"])
                    for s, f in zip(doc["spans"], doc["feats"])) / len(doc["spans"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep["ent"].append(ent.item())
        n = max(len(ep["r"]), 1)
        row = {"epoch": epoch, "r": round(sum(ep["r"]) / n, 4),
               "ph": round(sum(ep["ph"]) / n, 4),
               "ent": round(sum(ep["ent"]) / max(len(ep["ent"]), 1), 4),
               "ties_skipped": ep["ties_skipped"]}
        rows.append(row)
        if log_rows is not None:
            log_rows.append(row)
        print(f"[rt] epoch {epoch}: " +
              " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"), flush=True)
    return rows
```

Note on the patchable import: the test monkeypatches `tr.roundtrip_batch`, so bind it as a
module-level name — at the top of the file add
`from cloak.train.roundtrip import roundtrip_batch` guarded in a `try/except ImportError`
(surrogate-only environments without the module must still run), and inside `train_roundtrip`
use the module-level `roundtrip_batch` (drop the local `_rt` import — the code block above
shows the intent; final code uses the module-level name so tests can patch it).

3d. Wire `main()`: add the new CLI args; in roundtrip mode — load
`data/probes_validated.json` (path from `--probes`), override each doc's `probes_train` with
the validated train split, drop docs with < 3 (print the dropped count), set the mode
defaults (kl 0.0, adv rloo, entropy 0.01) unless flags were passed explicitly (use
`parser.get_default` comparison), run BC + `train_roundtrip` ONCE (no alpha loop), then the
greedy read-out (reuse the existing block but score `r` via one `roundtrip_batch` call over
greedy rollouts), and save `data/ranker_policy_rt.pt` + `results/ranker_train_rt.json` with
the same log schema plus `"reward": "roundtrip"`, `"rt_model"` from
`cloak.train.roundtrip.RT_MODEL`, `"adv": "rloo"`, `"ties_skipped"` per epoch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_train_roundtrip_mode.py -v`
Expected: 3 passed.

- [ ] **Step 5: Surrogate regression smoke (behavior must be byte-identical)**

Run: `PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_ranker.py --smoke --alphas 0.5`
Expected: completes as before (2 docs, BC reproduction / teacher checks print, epochs run);
`results/ranker_train_a0.5_smoke.json` written. If the GPU is busy (`pgrep -af train_pii`),
prepend `CUDA_VISIBLE_DEVICES=` to force CPU. Record the printed epoch rows in the report.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_ranker.py src/cloak/tests/test_train_roundtrip_mode.py
git commit -m "feat: trainer round-trip mode — RLOO advantage, tie filter, entropy bonus"
```

---

### Task 5: Trainer — expert-iteration (ExIt) outer loop

**Files:**
- Modify: `scripts/train_ranker.py`
- Test: `src/cloak/tests/test_exit_loop.py`

**Interfaces:**
- Consumes: `sample_rollout`, `roundtrip_batch` (module-level, patchable), `derive_spans`.
- Produces: `exit_round(docs, policy, *, G, rt_workers, seed) -> (winners, stats)` where
  winners = `[(doc_index, {surface: action_index})]` and stats =
  `{"mean_best_r": float, "mean_bc_r": float, "n_winners": int}`;
  `clone_choices(policy, items, epochs, lr)` where items =
  `[(spans, feats, {surface_lower: action_idx})]`; CLI `--exit-rounds INT` (default 0 = off),
  `--exit-epochs INT` (default 10). ExIt requires `--reward roundtrip` (assert).

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_exit_loop.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def test_exit_round_selects_winner_and_clones(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc()
    policy = tr.RankerPolicy()
    winners, stats = tr.exit_round([doc], policy, G=6, rt_workers=1, seed=0)
    # fake reward pays 1.0 for the level fill; the winner must have chosen action 0
    assert stats["n_winners"] == 1 and stats["mean_best_r"] == 1.0
    (di, choice_idx), = winners
    assert di == 0 and choice_idx["metformin"] == 0
    before = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach().clone()
    tr.clone_choices(policy, [(doc["spans"], doc["feats"], choice_idx)], epochs=20, lr=0.05)
    after = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach()
    assert after[0] > before[0]     # SFT on the winner raises the winning action's logp


def test_exit_round_no_winner_when_bc_optimal(monkeypatch):
    # reward everything 0 -> nothing beats the BC baseline -> no winners
    monkeypatch.setattr(tr, "roundtrip_batch",
                        lambda jobs, workers=1: [{"out_p": "", "out_final": "", "f1s": [0.0],
                                                  "recall": 0.0} for _ in jobs])
    torch.manual_seed(0)
    doc = _doc()
    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=4, rt_workers=1, seed=0)
    assert winners == [] and stats["n_winners"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_exit_loop.py -v`
Expected: FAIL with `AttributeError: exit_round`.

- [ ] **Step 3: Implement in `scripts/train_ranker.py`**

```python
def _bc_choice_indices(doc) -> dict[str, int]:
    return {s["surface"].lower(): s["bc_action"] for s in doc["spans"]}


def exit_round(docs, policy, *, G, rt_workers, seed):
    """One expert-iteration round (spec Phase 2 workhorse): per doc sample G rollouts,
    keep the best strictly beating the floor-walk baseline. Baselines and rollouts all go
    through the cached round trip. Returns (winners, stats)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    jobs, meta = [], []          # baseline job per doc first, then G rollouts per doc
    per_doc_idx = []
    for di, doc in enumerate(docs):
        bc_choice = {s["surface"].lower(): s["actions"][s["bc_action"]]
                     for s in doc["spans"]}
        try:
            doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], bc_choice)
            jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                         "probes": doc["probes_train"]})
            meta.append(("bc", di, None))
        except AssertionError:   # non-injective static teacher: baseline = -inf
            meta.append(("bc_skip", di, None))
        idxs = []
        for _ in range(G):
            choice, _, _, doc_p, R = sample_rollout(doc, doc["spans"], doc["feats"], policy)
            idx = {s["surface"].lower(): next(
                       i for i, a in enumerate(s["actions"])
                       if a is choice[s["surface"].lower()])
                   for s in doc["spans"]}
            jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                         "probes": doc["probes_train"]})
            meta.append(("roll", di, idx))
            idxs.append(idx)
        per_doc_idx.append(idxs)
    res = roundtrip_batch([j for j in jobs], workers=rt_workers)
    it = iter(res)
    bc_r, rolls = {}, {di: [] for di in range(len(docs))}
    for kind, di, idx in meta:
        if kind == "bc_skip":
            bc_r[di] = float("-inf")
            continue
        r = next(it)["recall"] or 0.0
        if kind == "bc":
            bc_r[di] = r
        else:
            rolls[di].append((r, idx))
    winners, best_rs = [], []
    for di in range(len(docs)):
        if not rolls[di]:
            continue
        best_r, best_idx = max(rolls[di], key=lambda t: t[0])
        best_rs.append(best_r)
        if best_r > bc_r[di]:
            winners.append((di, best_idx))
    stats = {"mean_best_r": round(sum(best_rs) / max(len(best_rs), 1), 4),
             "mean_bc_r": round(sum(v for v in bc_r.values() if v != float("-inf"))
                                / max(sum(v != float("-inf") for v in bc_r.values()), 1), 4),
             "n_winners": len(winners)}
    return winners, stats


def clone_choices(policy, items, epochs, lr):
    """SFT on winner action indices — behavior_clone generalized to arbitrary teachers."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        for spans, feats, choice_idx in items:
            loss = 0.0
            for s, f in zip(spans, feats):
                a_idx = choice_idx[s["surface"].lower()]
                if a_idx not in s["legal"]:
                    continue
                lp = policy.log_probs(f, s["legal"])
                loss = loss - lp[s["legal"].index(a_idx)]
            if isinstance(loss, torch.Tensor):
                opt.zero_grad()
                loss.backward()
                opt.step()
    return policy
```

Wire `main()`: when `--exit-rounds N > 0` (assert `args.reward == "roundtrip"`), after BC:
for each round call `exit_round`, then `clone_choices(policy, [(docs[di]["spans"],
docs[di]["feats"], idx) for di, idx in winners], epochs=args.exit_epochs, lr=args.lr)`,
log per-round stats into the run log (`"exit_rounds"` list), and only then run the optional
`train_roundtrip` refiner epochs (`--epochs 0` skips). Greedy read-out unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_exit_loop.py src/cloak/tests/test_train_roundtrip_mode.py -v`
Expected: 5 passed (Task 4's tests must still pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ranker.py src/cloak/tests/test_exit_loop.py
git commit -m "feat: expert-iteration outer loop (sample G, SFT on round-trip winners)"
```

---

### Task 6: Trainer — exact per-span counterfactual credit

**Files:**
- Modify: `scripts/train_ranker.py`
- Test: `src/cloak/tests/test_counterfactual_credit.py`

**Interfaces:**
- Consumes: `sample_rollout`, `roundtrip_batch`, `rloo_advantage`.
- Produces: `counterfactual_terms(doc, policy, choice, logps, base_r, *, frac, rng,
  rt_workers) -> (loss_term, n_cf)` — extra PG term from per-span placeholder
  counterfactuals; CLI `--cf-frac FLOAT` (default 0.0 = off; roundtrip mode only).

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_counterfactual_credit.py
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def test_counterfactual_credits_the_flipped_span(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc()
    policy = tr.RankerPolicy()
    # a rollout that KEPT the level fill (reward 1.0); counterfactual placeholder -> 0.0
    choice, logps, _, doc_p, _ = tr.sample_rollout(doc, doc["spans"], doc["feats"], policy,
                                                   greedy=True)
    if choice["metformin"]["mode"] != "level":   # force the level action for determinism
        choice = {"metformin": doc["spans"][0]["actions"][0]}
        lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
        logps = [lp[doc["spans"][0]["legal"].index(0)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=1.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 1
    # adv_span = base_r - r_cf = 1.0 - 0.0 = 1.0; term = -(adv * logp) > 0
    assert term.item() > 0
    term.backward()   # gradient flows to the policy
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_counterfactual_skips_placeholder_spans(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    policy = tr.RankerPolicy()
    ph_action = doc["spans"][0]["actions"][1]
    choice = {"metformin": ph_action}
    lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
    logps = [lp[doc["spans"][0]["legal"].index(1)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=0.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 0 and term == 0.0   # placeholder IS the counterfactual; nothing to flip
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_counterfactual_credit.py -v`
Expected: FAIL with `AttributeError: counterfactual_terms`.

- [ ] **Step 3: Implement in `scripts/train_ranker.py`**

```python
def counterfactual_terms(doc, policy, choice, logps, base_r, *, frac, rng, rt_workers):
    """Exact per-span credit (spec Phase 2; COMA made exact by reward determinism):
    for a sampled fraction of non-placeholder spans, re-run the round trip with ONLY that
    span flipped to its placeholder; adv_s = base_r - r_cf weights that span's logp.
    Counterfactual doc_p's are cache-friendly (identical across epochs at fixed choices)."""
    cand = [i for i, s in enumerate(doc["spans"])
            if choice[s["surface"].lower()]["mode"] == "level"]
    take = [i for i in cand if rng.random() < frac]
    if not take:
        return 0.0, 0
    jobs = []
    for i in take:
        s = doc["spans"][i]
        cf = dict(choice)
        ph_idx = next(k for k, a in enumerate(s["actions"]) if a["mode"] == "placeholder")
        cf[s["surface"].lower()] = s["actions"][ph_idx]
        doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], cf)
        jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                     "probes": doc["probes_train"]})
    res = roundtrip_batch(jobs, workers=rt_workers)
    term = 0.0
    for i, r in zip(take, res):
        adv_s = base_r - (r["recall"] or 0.0)
        term = term - adv_s * logps[i]
    return term, len(take)
```

Wire into `train_roundtrip`: after the group update, when `cf_frac > 0`, run one greedy
`sample_rollout`, score it with one `roundtrip_batch` call (its recall = `base_r`), compute
`counterfactual_terms(..., frac=cf_frac, ...)`, and apply it as its own
`opt.zero_grad(); term.backward(); opt.step()` step when `n_cf > 0` (guard
`isinstance(term, torch.Tensor)`). Log `cf_used` per epoch. Thread `--cf-frac` through
`main()` (roundtrip mode only; assert 0 ≤ cf_frac ≤ 1).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_counterfactual_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_exit_loop.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ranker.py src/cloak/tests/test_counterfactual_credit.py
git commit -m "feat: exact per-span counterfactual credit for round-trip RLOO"
```

---

### Task 7: Encoder policy (doc-conditioned ranker)

**Files:**
- Modify: `src/cloak/train/ranker.py` (add `EncoderPolicy`, `span_context`)
- Modify: `scripts/train_ranker.py` (CLI `--policy {mlp,encoder}`, `--encoder-model`)
- Test: `src/cloak/tests/test_encoder_policy.py`

**Interfaces:**
- Consumes: `action_features` (N_FEAT 17), `RankerPolicy`'s `.sample/.log_probs` contract.
- Produces:
  - `span_context(text: str, start: int, window: int = 256) -> str` — the ±window char slice
    around the span, whitespace-normalized.
  - `EncoderPolicy(encoder_name: str = "answerdotai/ModernBERT-base", feat_dim: int = N_FEAT,
    hid: int = 128)` with the SAME `.sample(feats, legal, greedy=False)` / `.log_probs(feats,
    legal)` interface as `RankerPolicy`, plus `.set_context(ctx_emb: torch.Tensor)` called
    per span before sample/log_probs (ctx_emb shape `[enc_dim]`), and
    `.embed_contexts(texts: list[str]) -> torch.Tensor` (frozen encoder, no_grad, batched).
  - Trainer: with `--policy encoder`, per-doc span contexts are embedded ONCE at load and
    attached as `doc["ctx"]` (list of tensors, one per span); every sample/log_probs call
    site sets the span's context first. `--encoder-model` overrides the HF name (tests use a
    tiny model).

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_encoder_policy.py
"""EncoderPolicy contract — uses a tiny random HF encoder so the test stays fast/offline
(hf-internal-testing models are ~100KB and live in the shared HF cache)."""
import torch

from cloak.train.ranker import N_FEAT, EncoderPolicy, span_context

TINY = "hf-internal-testing/tiny-random-bert"


def test_span_context_windows():
    text = "A" * 300 + " metformin " + "B" * 300
    ctx = span_context(text, start=301, window=50)
    assert "metformin" in ctx and len(ctx) <= 120


def test_encoder_policy_contract():
    torch.manual_seed(0)
    pol = EncoderPolicy(encoder_name=TINY)
    ctx = pol.embed_contexts(["patient on metformin 500mg daily"])
    assert ctx.shape[0] == 1
    pol.set_context(ctx[0])
    feats = torch.randn(3, N_FEAT)
    legal = [0, 2]
    lp = pol.log_probs(feats, legal)
    assert lp.shape == (2,) and torch.isfinite(lp).all()
    assert abs(lp.exp().sum().item() - 1.0) < 1e-4
    a, alp = pol.sample(feats, legal, greedy=True)
    assert a in legal and torch.isfinite(alp)
    # frozen encoder: only head params require grad
    enc_trainable = [p for p in pol.encoder.parameters() if p.requires_grad]
    assert not enc_trainable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_encoder_policy.py -v`
Expected: FAIL with `ImportError: EncoderPolicy`.

- [ ] **Step 3: Implement in `src/cloak/train/ranker.py`**

```python
def span_context(text: str, start: int, window: int = 256) -> str:
    """±window chars around the span start, whitespace-normalized — the encoder's view."""
    lo, hi = max(0, start - window), min(len(text), start + window)
    return " ".join(text[lo:hi].split())


class EncoderPolicy(torch.nn.Module):
    """Doc-conditioned ranker policy: score(action) = MLP([ctx_emb ; action_feats]).
    The encoder is FROZEN (feature extractor; embeddings precomputed per span at load) —
    only the head trains, so optimization cost matches the MLP policy.
    ponytail: no fine-tuning path; unfreeze via a separate task if capacity still binds."""

    def __init__(self, encoder_name: str = "answerdotai/ModernBERT-base",
                 feat_dim: int = N_FEAT, hid: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        enc_dim = self.encoder.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(enc_dim + feat_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1))
        self._ctx = None

    @torch.no_grad()
    def embed_contexts(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
        enc = {k: v.to(next(self.head.parameters()).device) for k, v in enc.items()}
        return self.encoder(**enc).last_hidden_state[:, 0]      # CLS per text

    def set_context(self, ctx_emb: torch.Tensor):
        self._ctx = ctx_emb

    def log_probs(self, feats: torch.Tensor, legal: list[int]) -> torch.Tensor:
        assert self._ctx is not None, "call set_context(ctx_emb) before scoring"
        ctx = self._ctx.unsqueeze(0).expand(len(legal), -1)
        x = torch.cat([ctx, feats[legal]], dim=-1)
        return torch.log_softmax(self.head(x).squeeze(-1), dim=-1)

    def sample(self, feats: torch.Tensor, legal: list[int], greedy: bool = False):
        lp = self.log_probs(feats, legal)
        j = int(lp.argmax()) if greedy else int(torch.multinomial(lp.exp(), 1))
        return legal[j], lp[j]
```

Match `RankerPolicy`'s exact sample/log_probs semantics (compare with the existing class and
mirror any detail this sketch missed — e.g. device handling, dtype). In
`scripts/train_ranker.py`: `--policy {mlp,encoder}` (default mlp), `--encoder-model`
(default `answerdotai/ModernBERT-base`); when encoder — build the policy once, precompute
`doc["ctx"] = policy.embed_contexts([span_context(doc["text"], s["start"]) for s in
doc["spans"]])` at doc load, and wrap every `policy.sample/log_probs/policy_entropy/
kl_to_ref` call site to `policy.set_context(doc["ctx"][i])` for span i first (BC, RL,
ExIt, counterfactual paths all included — grep every `\.sample\(|log_probs\(` call site).
The reference policy for KL in encoder mode is a second `EncoderPolicy` sharing the SAME
frozen encoder object but a deep-copied head (implement `clone_for_ref()` returning that).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_encoder_policy.py -v`
Expected: 2 passed. Then the full trainer test set:
`PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests -k "train or exit or counterfactual or encoder" -v`
Expected: all pass (mlp default untouched).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/ranker.py scripts/train_ranker.py src/cloak/tests/test_encoder_policy.py
git commit -m "feat: frozen-encoder doc-conditioned ranker policy (--policy encoder)"
```

---

### Task 8: Pilot scale parameters + LFM saturation probe

**Files:**
- Modify: `scripts/train_ranker.py`, `scripts/reward_gate.py`, `scripts/build_probes.py`,
  `scripts/spikes/roundtrip_support_scan.py` (thread `--n-docs`; replace every hardcoded
  `load_task_docs(corpus, 16)`)
- Create: `scripts/spikes/lfm_saturation_probe.py`
- Test: `src/cloak/tests/test_n_docs_arg.py`

**Interfaces:**
- Produces: `--n-docs INT` (default 16 — current behavior preserved) on all four scripts;
  `scripts/spikes/lfm_saturation_probe.py` printing
  `{"workers": w, "n": n, "wall_s": ..., "rt_per_hour": ..., "approx_tok_s": ...}` per
  worker setting.

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_n_docs_arg.py
"""--n-docs is threaded (no hardcoded 16 left on the doc-loading paths)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ["scripts/train_ranker.py", "scripts/reward_gate.py", "scripts/build_probes.py",
           "scripts/spikes/roundtrip_support_scan.py"]


def test_no_hardcoded_doc_count():
    for s in SCRIPTS:
        src = (ROOT / s).read_text()
        assert not re.search(r"load_task_docs\([^)]*,\s*16\s*\)", src), s
        assert "--n-docs" in src, s


def test_help_exposes_n_docs():
    for s in ["scripts/train_ranker.py"]:
        out = subprocess.run([sys.executable, str(ROOT / s), "--help"],
                             capture_output=True, text=True,
                             env={"PYTHONPATH": f"{ROOT}/src:{ROOT}/scripts", "PATH": "/usr/bin:/bin"})
        assert "--n-docs" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_n_docs_arg.py -v`
Expected: FAIL (hardcoded 16s present).

- [ ] **Step 3: Thread `--n-docs` + write the saturation probe**

In each script: `ap.add_argument("--n-docs", type=int, default=16)` and pass `args.n_docs`
to every `load_task_docs` call. (Env/artifact coverage stays the binding constraint — docs
beyond the artifact are skipped by the existing `d["id"] in per_doc` guards; note this in
the `--n-docs` help string: "docs beyond the frozen arms artifact are skipped".)

```python
# scripts/spikes/lfm_saturation_probe.py
"""10-minute saturation probe for the pinned round-trip model (perf gate prerequisite:
the plan's wall-time estimates assume ~1500 RT/h — this measures the real number).

Unique prompts (cache-busting nonce) through the REAL task template at workers 1 and 6.
Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/lfm_saturation_probe.py \
       [--n 24] [--workers 1,6]     # NO cache env var — this must hit the model
"""
import argparse
import json
import time
import uuid

from cloak.corpora import load_task_docs
from cloak.tasks import TASK_TEMPLATE
from cloak.train.roundtrip import MAX_TOKENS, RT_MODEL
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", default="1,6")
    args = ap.parse_args()
    docs = load_task_docs("clinical", max(4, args.n // 4))
    remote = LLMClient(RT_MODEL, temperature=0.0, max_tokens=MAX_TOKENS,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for w in [int(x) for x in args.workers.split(",")]:
        prompts = [TASK_TEMPLATE["clinical"].format(doc=docs[i % len(docs)]["text"])
                   + f"\n[probe-nonce {uuid.uuid4()}]" for i in range(args.n)]
        t0 = time.time()
        outs = pmap(remote.generate, prompts, workers=w)
        wall = time.time() - t0
        toks = sum(len(o) for o in outs) / 4          # chars/4 ~ tokens
        print(json.dumps({"workers": w, "n": args.n, "wall_s": round(wall, 1),
                          "rt_per_hour": round(args.n / wall * 3600),
                          "approx_tok_s": round(toks / wall)}), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_n_docs_arg.py -v`
Expected: 2 passed. Also re-run the full new-test suite:
`PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests -v` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ranker.py scripts/reward_gate.py scripts/build_probes.py \
        scripts/spikes/roundtrip_support_scan.py scripts/spikes/lfm_saturation_probe.py \
        src/cloak/tests/test_n_docs_arg.py
git commit -m "feat: --n-docs threading + LFM saturation probe spike"
```

---

## Runbook (gated — NOT tasks; run only after all tasks review-clean)

Ordered; each step gates the next. All long runs: `-u`, logged to `results/`, GPU checked
(`pgrep -af train_pii`) first. The pilot RUN itself additionally requires the perf gate
(`/auto-review-loop` vs `scripts/harness/perf_gate.md`) before launch.

1. `lfm_saturation_probe` (~10 min, no cache) → real RT/h; recompute wall-time plan.
2. `build_probes.py --n-docs 16` on the EXISTING env/artifact (teacher phase first — gemma;
   then LFM anchors) → `results/probe_health.json`; check ceiling pass rate (the LFM
   go/no-go from the spec table) and excluded-doc counts.
3. `roundtrip_support_scan.py --max-swaps 150` (~30 min) → **verdict PASS required before
   any training run** (handoff-mandated gate).
4. First-smoke: `train_ranker.py --reward roundtrip --smoke` (2 docs) — movement canary.
5. Pilot training run (fixed floors first, then `--randomize-floors`; ExIt then refiner) —
   spec-then-results training record `research-wiki/training/2026-07-0X-RL-ranker-v3-roundtrip-pilot.md`
   written BEFORE the run (v-schema).
6. Scale `--n-docs` only after 1–5 hold on the 16-doc environment (a bigger env needs a NEW
   frozen arms artifact build — separate decision, separate record).

## Out of scope (explicitly)

- π_fill / E1 / grammar artifacts (Stage 2; binding TBD per user decision 2026-07-05).
- New arms artifact / detection runs / corpus downloads (Multi-LexSum etc.) — runbook-gated
  environment builds, not code tasks.
- Distilled RM screening (needs a populated cache first; add after the pilot produces one).
- Whole-task-quality regression gate at eval (Phase-5 machinery; next plan).
