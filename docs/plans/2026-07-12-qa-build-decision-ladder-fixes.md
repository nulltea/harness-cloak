---
type: plan
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [rl, qa-build, ladder-probes, decision-probes, reward, implementation-plan]
companion: [docs/issues/2026-07-11-ladder-decision-qa-question-design.md]
---

# QA-build decision/ladder fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mechanical defects that make the decision probe tier unable to discriminate placeholder-vs-preserve and make ladder rung questions redundant, per the issue register `docs/issues/2026-07-11-ladder-decision-qa-question-design.md` and the 2026-07-12 code analysis.

**Architecture:** Six code changes on the existing QA-build pipeline: (1) a dedicated multiple-choice reader prompt beside the extractive QA prompt in `reward.py`; (2) binary containment scoring for the semantic acceptance set (`entail_score`); (3) decision scoring in the reward reads `out_p` (pre-inversion) instead of `out_final`; (4) decision validation reads `out_p`, uses one option shuffle for both anchors, and rejects decisions not linked to a detected span; (5) a decision leak lint + `DECISION_PROMPT` rewrite that forbids naming the target fact (`DECISION_PV` 3→4); (6) the ladder teacher asks exactly two questions per span — rung 0 (exact) and rung 1 (finest generalization) — instead of one per rung (`LADDER_PV` 3→4). Then a controller-run sweep rerun regenerates the validated artifacts.

**Tech Stack:** Python 3.11, pytest, existing repo modules only (no new dependencies).

## Global Constraints

- Work on branch `ladder-build-validate` (already checked out). Never touch `main`.
- **Stray-file guard (shared checkout, parallel sessions):** the working tree has pre-existing uncommitted changes NOT belonging to this plan: `data/lattice_profiles/lattice_profiles.json`, `scripts/run_lattice_producer.py`, `src/cloak/anonymity.py`, `src/cloak/tests/test_build_mined_lattice_profiles.py`. NEVER `git add -A`/`git add .`. Stage only the exact files your task names, then run `git diff --cached --name-only` and confirm it lists ONLY your files before committing.
- Test command (offline, no proxy needed — all reader/teacher calls are monkeypatched): `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_mc_reader.py -q` (the third file exists after Task 1; before that, omit it).
- Module self-check after editing `src/cloak/train/ladder_probes.py`: `PYTHONPATH=src .venv/bin/python src/cloak/train/ladder_probes.py` must print `ladder_probes.py self-check OK`.
- Exact pin values: `LADDER_PV = 4` (Task 6), `DECISION_PV = 4` (Task 5). No other pin changes (`QA_MODEL`, `RT_MODEL`, `TH`, `W_EXACT`, `W_SEM` stay).
- Naming rule: name code after behavior (`lint_decision`, `_category_hit`), never after issue/plan numbering.
- Match existing module style: compact multi-line docstrings that state the WHY (see `entail_score`, `_reusable` for the register style).
- End every commit message with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Multiple-choice reader prompt in `reward.py`

The pinned reader (`Qwen3.5-0.8B`) answers everything through `QA_PROMPT`, which demands "the shortest exact answer **copied from the note** … else NONE". Multiple-choice decision questions have gold options ("Cardiologist") that are rarely verbatim in the note, so an obedient reader answers NONE and good decisions ceiling-reject. Add an MC template beside the extractive one, plus the shared option-block renderer (moved here from `roundtrip.py` in Task 3, where a private copy currently lives).

**Files:**
- Modify: `src/cloak/train/reward.py` (around lines 28–76: `QA_PROMPT`, `_read_batch`)
- Test: `src/cloak/tests/test_mc_reader.py` (create)

**Interfaces:**
- Consumes: existing `_qa_client()`, `_parse()` in `reward.py`.
- Produces: `MC_PROMPT` (module constant), `decision_prompt(q: str, options: list[str]) -> str`, `_read_mc_batch(questions: list[str], context: str, refresh: bool = False) -> list[str]`. Tasks 3 and 4 import all three.

- [ ] **Step 1: Write the failing test**

Create `src/cloak/tests/test_mc_reader.py`:

```python
"""MC reader template: decisions must not go through the extractive QA prompt."""

import cloak.train.reward as rw


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, refresh=False):
        self.prompts.append(prompt)
        return self.reply


def test_read_mc_batch_uses_mc_template_not_extractive_qa(monkeypatch):
    fake = FakeClient("route to endocrinology")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)

    q = rw.decision_prompt("Which route?", ["primary care", "route to endocrinology"])
    out = rw._read_mc_batch([q], "NOTE TEXT")

    assert out == ["route to endocrinology"]
    p = fake.prompts[0]
    assert "NOTE TEXT" in p and "Which route?" in p and "- primary care" in p
    # the extractive framing that made non-verbatim options unanswerable must be gone
    assert "copied from the note" not in p
    assert "NONE" not in p


def test_read_batch_still_uses_extractive_template(monkeypatch):
    fake = FakeClient("hypothyroidism")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)

    out = rw._read_batch(["What diagnosis?"], "NOTE TEXT")

    assert out == ["hypothyroidism"]
    assert "copied from the note" in fake.prompts[0]


def test_read_mc_batch_empty_questions_and_none_reply(monkeypatch):
    assert rw._read_mc_batch([], "NOTE") == []
    fake = FakeClient("NONE")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)
    assert rw._read_mc_batch([rw.decision_prompt("Q?", ["a", "b"])], "NOTE") == [""]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_mc_reader.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'decision_prompt'` (or `_read_mc_batch`).

- [ ] **Step 3: Implement**

In `src/cloak/train/reward.py`, directly below the `QA_PROMPT` assignment, add:

```python
MC_PROMPT = ("Read the note, then answer the multiple-choice question. Choose the single best "
             "option supported by the note; the correct option's text may not appear verbatim "
             "in the note. Reply with the exact text of the chosen option and nothing else."
             "\n\nNote:\n{ctx}\n\n{q}\nAnswer:")
```

Replace the body of `_read_batch` with a shared template reader and add the MC variants (keep `_read_batch`'s docstring content on `_read_with`; the wrappers get one-liners):

```python
def _read_with(template: str, questions: list[str], context: str,
               refresh: bool = False) -> list[str]:
    """Grounded QA over ONE context via the served reader, issued SERIALLY (workers=1) so the
    questions hit one llama.cpp slot in sequence and its prompt-cache reuses the shared note-
    prefix KV (measured ~6.6x faster than fanning them across slots, which re-prefills the
    note per question; local prefix-KV is unavailable — Qwen3.5's hybrid cache breaks it).
    Cross-context parallelism belongs one level up (concurrent docs/rollouts -> the 6 slots).
    Deterministic greedy (temp0), non-thinking; '' on NONE."""
    if not questions:
        return []
    client = _qa_client()
    return [_parse(client.generate(template.format(ctx=context, q=q), refresh=refresh))
            for q in questions]


def _read_batch(questions: list[str], context: str, refresh: bool = False) -> list[str]:
    """Extractive short-answer reads (QA_PROMPT): the fact-recall channels."""
    return _read_with(QA_PROMPT, questions, context, refresh)


def _read_mc_batch(questions: list[str], context: str, refresh: bool = False) -> list[str]:
    """Multiple-choice reads (MC_PROMPT). Decisions' gold options are often not verbatim in
    the note; the extractive QA_PROMPT instructs 'copied from the note'/NONE and so
    ceiling-rejects exactly the note-dependent decisions the tier is for."""
    return _read_with(MC_PROMPT, questions, context, refresh)


def decision_prompt(q: str, options: list[str]) -> str:
    """Render an MC decision question + options block (the {q} slot of MC_PROMPT)."""
    return q + "\nOptions:\n" + "\n".join(f"- {o}" for o in options)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_mc_reader.py src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_ladder_probe_build.py -q`
Expected: all PASS (the existing suites don't touch the changed internals).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/reward.py src/cloak/tests/test_mc_reader.py
git diff --cached --name-only   # must list ONLY the two files above
git commit -m "feat(qa-build): dedicated MC reader prompt beside the extractive QA prompt

The extractive QA_PROMPT ('copied from the note'/NONE) ceiling-rejects decisions
whose gold option is not verbatim in the note. Decisions now read via MC_PROMPT.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Binary containment scoring for the semantic acceptance set

`entail_score` falls back to token-F1: sibling categories sharing a head noun ("vascular disease" vs gold "artery disease") score exactly 0.5 — precisely at `TH = 0.5` — so the rung cutoff sits on a knife edge and coarser fills earn partial credit. Make the semantic-tier match binary: full content-token containment (articles stripped), acronym, or zero.

**Files:**
- Modify: `src/cloak/train/ladder_probes.py:196-201` (`entail_score`) and the `__main__` self-check block
- Test: `src/cloak/tests/test_ladder_probe_build.py` (replace `test_validate_ladder_threshold_boundaries_are_inclusive_for_hi_strict_for_lo`, add sibling-category test)

**Interfaces:**
- Consumes: existing `fact_score`, `_tokens` in the same module.
- Produces: `entail_score(answer, rungs, rung, aliases=())` — same signature, now returns only 0.0 or 1.0. `_score_ladder` (roundtrip) and `validate_ladder` pick the change up automatically. The rung-0 echo channel in `_score_ladder` (`fact_score` direct) is intentionally untouched — echo stays graded.

- [ ] **Step 1: Update the tests**

In `src/cloak/tests/test_ladder_probe_build.py`, DELETE `test_validate_ladder_threshold_boundaries_are_inclusive_for_hi_strict_for_lo` entirely (its 0.5-boundary semantics are the defect being removed) and add in its place:

```python
def test_entail_score_is_binary_no_shared_head_noun_credit():
    from cloak.train.ladder_probes import entail_score

    rungs = ["hypertension", "artery disease", "vascular disease"]
    # sibling category sharing only the head noun 'disease' must score 0, not F1 0.5
    assert entail_score("vascular disease", rungs, 1) == 0.0
    # finer-than-gold answer containing the gold's content tokens is a full hit
    assert entail_score("coronary artery disease", rungs, 1) == 1.0
    # article difference must not break containment
    assert entail_score("cardiovascular disease",
                        ["heart failure", "a cardiovascular disease"], 1) == 1.0
    # acronym path survives (fact_score exact hit)
    assert entail_score("CHF", ["congestive heart failure"], 0) == 1.0


def test_validate_ladder_partial_token_answers_reject_not_partial_score():
    entries = [
        {
            "id": "partial-hi",
            "surface": "alpha beta gamma",
            "rungs": ["alpha beta gamma"],
            "rung": 0,
            "q": "What is the first boundary fact?",
        },
        {
            "id": "partial-lo",
            "surface": "delta epsilon zeta",
            "rungs": ["delta epsilon zeta"],
            "rung": 0,
            "q": "What is the second boundary fact?",
        },
    ]
    hi = {
        entries[0]["q"]: "alpha",                    # one shared token -> 0.0 -> ceiling
        entries[1]["q"]: "delta epsilon zeta noted",  # containment -> 1.0
    }
    lo = {
        entries[0]["q"]: "",
        entries[1]["q"]: "delta",                    # one shared token -> 0.0 -> NOT floor
    }

    kept, rows = validate_ladder(entries, hi.get, hi.get, lo.get, lo.get, th=0.5)

    assert [e["id"] for e in kept] == ["partial-lo"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "partial-hi": "ceiling",
        "partial-lo": "kept",
    }
    assert all(r["hi_score"] in (0.0, 1.0) and r["lo_score"] in (0.0, 1.0) for r in rows)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py -q`
Expected: the two new tests FAIL (`entail_score("vascular disease", ...) == 0.5`, not 0.0); the rest pass.

- [ ] **Step 3: Implement**

In `src/cloak/train/ladder_probes.py`, replace `entail_score` with:

```python
def _category_hit(answer: str, gold: str) -> float:
    """Binary semantic-tier match: fact_score full hit (exact/containment/acronym) or content-
    token containment with articles/stopwords stripped. NO token-F1 partial credit — sibling
    categories share head nouns ('vascular disease' vs 'artery disease' both carry 'disease',
    F1 0.5 = exactly TH), which blurs the rung cutoff the specificity gradient depends on."""
    if fact_score(answer, gold) == 1.0:
        return 1.0
    gold_t = _tokens(gold)
    return 1.0 if gold_t and gold_t <= _tokens(answer) else 0.0


def entail_score(answer: str, rungs: list[str], rung: int, aliases=()) -> float:
    """Acceptance-set scoring: an answer at or finer than the rung counts (binary, via
    _category_hit). `aliases` are surface-equivalent strings (the matched canonical + its
    profile aliases); they are the finest tier, so they satisfy every rung — folding them in
    accepts synonym answers the note may use (HTN vs hypertension) that the exact surface
    alone would miss."""
    return max(_category_hit(answer, a) for a in [*rungs[: rung + 1], *aliases])
```

In the `__main__` self-check block, after the existing `entail_score` asserts, add:

```python
    # binary semantic match: sibling category sharing a head noun scores 0, never F1 0.5
    assert entail_score("vascular disease", ["hypertension", "artery disease"], 1) == 0.0
    assert entail_score("coronary artery disease", ["hypertension", "artery disease"], 1) == 1.0
```

- [ ] **Step 4: Run tests + self-check to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py src/cloak/tests/test_two_channel_reward.py -q && PYTHONPATH=src .venv/bin/python src/cloak/train/ladder_probes.py`
Expected: all PASS; self-check prints `ladder_probes.py self-check OK`.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/ladder_probes.py src/cloak/tests/test_ladder_probe_build.py
git diff --cached --name-only   # must list ONLY the two files above
git commit -m "fix(qa-build): binary containment scoring for semantic acceptance sets

Token-F1 fallback gave sibling categories sharing a head noun exactly 0.5 = TH,
blurring the rung cutoff. entail_score is now containment/acronym-or-zero.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Reward path — score decisions on `out_p` via the MC reader

`_score_decisions` reads `out_final`, where `invert()` restores echoed placeholders (verified: the floor anchor's `out_final` contains "congestive heart failure"). A placeholder-everything policy therefore still earns decision credit whenever the placeholder echoes. Read decisions on `out_p` (pre-inversion, like semantic rungs) through the MC template from Task 1.

**Files:**
- Modify: `src/cloak/train/roundtrip.py` (imports at lines 19–22; `_decision_prompt`/`_score_decisions` at lines 107–124; the `_one` call site at line 182)
- Test: `src/cloak/tests/test_two_channel_reward.py`

**Interfaces:**
- Consumes: `_read_mc_batch`, `decision_prompt` from `cloak.train.reward` (Task 1).
- Produces: `_score_decisions(decisions, out_p, refresh)` — decision channel reads pre-inversion output. `roundtrip_batch` result shape unchanged.

- [ ] **Step 1: Update the tests**

In `src/cloak/tests/test_two_channel_reward.py`:

1. In `test_carrier_combines_available_components_unweighted`, decision reads now flow through `rt._read_mc_batch`. Replace the single `fake_read` + monkeypatch with two fakes and assert the decision context is the pre-inversion output:

```python
def test_carrier_combines_available_components_unweighted(monkeypatch):
    stub = _StubClient(["OUT_P: patient has <CONDITION_1>.",
                        "OUT_P: patient has <CONDITION_1>."])
    decision_reads = []
    shuffle_seed_keys = []

    def fake_read(questions, context, refresh=False):
        return ["hypothyroidism" if q.startswith("What exact") else "" for q in questions]

    def fake_read_mc(questions, context, refresh=False):
        # ONE batched call carries both decision prompts -> answer per-question by content
        decision_reads.extend((q, context) for q in questions)
        return ["route to endocrinology" if "route" in q else "primary care"
                for q in questions]

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)
    monkeypatch.setattr(rt, "_read_mc_batch", fake_read_mc)
    monkeypatch.setattr(
        rt,
        "mc_shuffle",
        lambda options, seed_key: shuffle_seed_keys.append(seed_key) or list(reversed(options)),
    )
    monkeypatch.setattr(rt, "schema_field_score", lambda out_final, out_hi: 1.0)
    decisions = [
        {"q": "Which route?", "options": ["primary care", "route to endocrinology"],
         "gold": "route to endocrinology", "span_ids": ["s0"]},
        {"q": "What follow-up?", "options": ["primary care", "cardiology"],
         "gold": "cardiology", "span_ids": ["s0"]},
    ]

    res = rt.roundtrip_batch([
        _job(decisions=decisions, schema=True, out_hi="CEILING")
    ], workers=1)[0]

    assert res["decision_score"] == pytest.approx(0.5)
    assert res["schema_score"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert len(shuffle_seed_keys) == 2
    assert shuffle_seed_keys[0] != shuffle_seed_keys[1]
    # decisions must read the PRE-inversion output: invert() restores echoed placeholders
    # into out_final, so out_final cannot discriminate placeholder-vs-preserve
    assert all(ctx.startswith("OUT_P") for _q, ctx in decision_reads)
    assert all("Options:" in q for q, _ctx in decision_reads)
```

2. Replace `test_main_reward_scores_span_free_decisions` with:

```python
def test_main_reward_scores_span_free_decisions(monkeypatch):
    stub = _StubClient(["OUT_P: patient has <CONDITION_1>."])
    decision_prompts = []

    def fake_read_mc(questions, context, refresh=False):
        decision_prompts.extend(questions)
        return ["route to endocrinology" if "route" in q else "routine" for q in questions]

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL", None))
    monkeypatch.setattr(rt, "_read_batch",
                        lambda questions, context, refresh=False: [""] * len(questions))
    monkeypatch.setattr(rt, "_read_mc_batch", fake_read_mc)

    decisions = [
        {"q": "Which route?", "options": ["primary care", "route to endocrinology"],
         "gold": "route to endocrinology", "span_ids": ["s0"]},
        {"q": "Which billing path?", "options": ["routine", "complex"],
         "gold": "routine", "span_ids": []},
    ]

    res = rt.roundtrip_batch([
        _job(ladder=[], decisions=decisions)
    ], workers=1)[0]

    assert res["decision_score"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert len(decision_prompts) == 2
    assert all("Options:" in p for p in decision_prompts)
```

3. Replace `test_reader_refresh_reaches_echo_semantic_and_decision_channels` with:

```python
def test_reader_refresh_reaches_echo_semantic_and_decision_channels(monkeypatch):
    stub = _StubClient(["OUT_P: patient has an endocrine condition."])
    refreshes = []

    def fake_read(questions, context, refresh=False):
        refreshes.extend(refresh for _ in questions)
        return ["hypothyroidism" if "exact" in q else "an endocrine condition"
                for q in questions]

    def fake_read_mc(questions, context, refresh=False):
        refreshes.extend(refresh for _ in questions)
        return ["endocrinology"] * len(questions)

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)
    monkeypatch.setattr(rt, "_read_mc_batch", fake_read_mc)

    rt.roundtrip_batch([
        _job(decisions=[{"q": "Which route?", "options": ["endocrinology", "primary care"],
                         "gold": "endocrinology"}])
    ], workers=1, reader_refresh=True)

    assert refreshes == [True, True, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_two_channel_reward.py -q`
Expected: the three updated tests FAIL (`rt` has no `_read_mc_batch`; decision reads still hit `_read_batch` with `OUT_FINAL` context).

- [ ] **Step 3: Implement**

In `src/cloak/train/roundtrip.py`:

1. Extend the reward import to:

```python
from cloak.train.reward import (_max_by_fact, _read_batch, _read_mc_batch, canon,
                                decision_prompt, fact_f1s, fact_score, mc_score,
                                W_EXACT, W_SEM)
```

2. Delete the local `_decision_prompt` function.

3. Replace `_score_decisions` with:

```python
def _score_decisions(decisions: list[dict], out_p: str, refresh: bool) -> float | None:
    """Decision channel reads OUT_P (pre-inversion), like the semantic rungs: invert()
    restores echoed placeholders into out_final, so a placeholder-everything policy would
    still earn decision credit there whenever the placeholder echoes. On out_p a
    placeholdered fact is a literal '<TYPE_N>' token and the decision breaks, as the tier
    intends."""
    rows = []
    for i, entry in enumerate(_kept(decisions)):
        q, options, gold = entry.get("q"), entry.get("options") or [], entry.get("gold")
        if not q or not options or gold is None:
            continue
        shuffled = mc_shuffle(options, f"{q}|{i}|{out_p}")
        rows.append((q, shuffled, gold))
    if not rows:
        return None
    answers = _read_mc_batch([decision_prompt(q, opts) for q, opts, _ in rows],
                             out_p, refresh=refresh)
    scores = [mc_score(answer, gold, opts) for answer, (_q, opts, gold) in zip(answers, rows)]
    return sum(scores) / len(scores)
```

4. In `_one`, change the call site to pass the pre-inversion output:

```python
            decision_score = _score_decisions(j.get("decisions") or [], op,
                                              reader_refresh)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_mc_reader.py src/cloak/tests/test_ladder_probe_build.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/roundtrip.py src/cloak/tests/test_two_channel_reward.py
git diff --cached --name-only   # must list ONLY the two files above
git commit -m "fix(qa-build): decision reward channel reads out_p via the MC reader

invert() restores echoed placeholders into out_final, so decisions scored there
never discriminate placeholder-vs-preserve. Decisions now read the pre-inversion
output through MC_PROMPT, like the semantic rungs.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Decision validation — `out_p` anchors, symmetric shuffle, span-link gate

Three defects in `validate_decisions` + its `build_probes.py` wiring: (a) hi/lo anchors are `out_final` (same echo-inversion problem as Task 3 — 6/7 decisions in the super sweep floor-rejected because the floor `out_final` had the fact names restored); (b) hi and lo use different option shuffles, so verdicts carry pure reader noise (the one kept decision survived only because the floor reader mis-picked under a different option order); (c) `span_ids` is computed but never gated on — a decision linked to no detected span can't be credited to a span and can't break under placeholdering.

**Files:**
- Modify: `src/cloak/train/ladder_probes.py:306-335` (`validate_decisions`)
- Modify: `scripts/build_probes.py:191-203` (`_reader_mc_for_context`), lines 294–298 (`build_ladder` decision validation call), lines 539–543 (`build_ladder_detected` decision validation call)
- Test: `src/cloak/tests/test_ladder_probe_build.py`

**Interfaces:**
- Consumes: `_read_mc_batch`, `decision_prompt` from Task 1; `_mc_pick` from `cloak.train.reward` (exists).
- Produces: `validate_decisions(entries, reader_mc_hi, reader_mc_lo)` — same signature; new verdict value `"unlinked"` (rejected, reader not called); both readers now receive the SAME shuffled option list. `_reader_mc_for_context(context)` unchanged signature.

- [ ] **Step 1: Update/add tests**

In `src/cloak/tests/test_ladder_probe_build.py`:

1. In `test_validate_decisions_tags_spans_from_depends_on_canon_substring`, entries `d2` and `d3` have `depends_on` matching no span → both now reject as `unlinked` (reader never consulted). Replace the final assertions with:

```python
    assert [e["id"] for e in kept] == ["d1"]
    assert kept[0]["span_ids"] == ["s-condition", "s-drug"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "d1": "kept",
        "d2": "unlinked",
        "d3": "unlinked",
    }
    assert rows[1]["hi_pick"] is None and rows[1]["lo_pick"] is None
```

2. Add:

```python
def test_validate_decisions_hi_and_lo_read_the_same_option_order():
    # hi/lo shuffles differed by seed suffix, so keep/floor verdicts carried option-order
    # noise on the positional-bias-prone small reader (measured: the one kept decision of
    # the 2026-07-12 super sweep survived only via a floor mis-pick under a different order)
    seen = {}
    entry = {
        "id": "d1",
        "q": "Which route is supported?",
        "options": ["primary care", "endocrinology", "cardiology", "neurology"],
        "gold": "endocrinology",
        "depends_on": ["hypothyroidism"],
        "detected_spans": [{"id": "s0", "surface": "hypothyroidism"}],
    }

    def hi(q, options):
        seen["hi"] = list(options)
        return "endocrinology"

    def lo(q, options):
        seen["lo"] = list(options)
        return "primary care"

    kept, rows = validate_decisions([entry], hi, lo)

    assert seen["hi"] == seen["lo"]
    assert rows[0]["verdict"] == "kept"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py -q`
Expected: the updated span-tag test FAILS (`d3` still kept) and the new shuffle test FAILS (`seen["hi"] != seen["lo"]`).

- [ ] **Step 3: Implement `validate_decisions`**

In `src/cloak/train/ladder_probes.py`, replace `validate_decisions` with:

```python
def validate_decisions(entries, reader_mc_hi, reader_mc_lo):
    """Validate multiple-choice decision probes with injected pinned readers.

    Readers must read the PRE-inversion anchors (out_p): invert() restores echoed
    placeholders into out_final, so out_final cannot discriminate placeholder-vs-preserve
    (measured 2026-07-12: floor out_final had all fact names restored -> 6/7 floor-rejects).
    Both readers get the SAME shuffled option order — differing orders let the small
    reader's positional bias masquerade as a context effect. A decision whose depends_on
    matches no detected span is rejected 'unlinked': it cannot be credited to a span and
    cannot break under placeholdering, so it carries no training signal."""
    kept, rows = [], []
    for idx, e in enumerate(entries):
        q = e.get("q", "")
        gold = e.get("gold")
        span_ids = _decision_span_ids(e)
        hi_pick = lo_pick = None
        if not span_ids:
            verdict = "unlinked"
        else:
            options = mc_shuffle(e.get("options") or [], f"{q}|{idx}")
            hi_pick = reader_mc_hi(q, options)
            lo_pick = reader_mc_lo(q, options)
            if hi_pick != gold:
                verdict = "ceiling"
            elif lo_pick == gold:
                verdict = "floor"
            else:
                verdict = "kept"
        row = {
            "id": e.get("id") or q,
            "q": q,
            "gold": gold,
            "hi_pick": hi_pick,
            "lo_pick": lo_pick,
            "span_ids": span_ids,
            "verdict": verdict,
        }
        rows.append(row)
        out = {**e, "span_ids": span_ids, "validation": row}
        if verdict == "kept":
            kept.append(out)
    return kept, rows
```

- [ ] **Step 4: Implement the `build_probes.py` wiring**

1. Replace `_reader_mc_for_context` with (reuses the shared option matcher instead of the inline copy):

```python
def _reader_mc_for_context(context):
    from cloak.train.reward import _mc_pick, _read_mc_batch, decision_prompt

    def read(q, options):
        return _mc_pick(_read_mc_batch([decision_prompt(q, options)], context)[0], options)
    return read
```

2. In `build_ladder` (currently lines 294–298), point the decision readers at the pre-inversion anchors:

```python
            kept_decisions, decision_rows = lp.validate_decisions(
                decision_entries,
                _reader_mc_for_context(anchor[doc_id]["hi"]["out_p"]),
                _reader_mc_for_context(anchor[doc_id]["lo"]["out_p"]),
            )
```

3. Same change in `build_ladder_detected` (currently lines 539–543).

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_mc_reader.py -q && PYTHONPATH=src .venv/bin/python src/cloak/train/ladder_probes.py`
Expected: all PASS + self-check OK.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/train/ladder_probes.py scripts/build_probes.py src/cloak/tests/test_ladder_probe_build.py
git diff --cached --name-only   # must list ONLY the three files above
git commit -m "fix(qa-build): decision validation on out_p, symmetric shuffle, span-link gate

- hi/lo decision anchors switch from out_final (echo-inversion restores fact
  names -> blanket floor-rejects) to out_p
- one option shuffle for both anchors (order noise masqueraded as context effect)
- decisions matching no detected span reject as 'unlinked'
- MC reads go through MC_PROMPT via the shared _mc_pick matcher

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Decision leak lint + `DECISION_PROMPT` no-naming rewrite (`DECISION_PV` 4)

The decision path has no leakage lint (the ladder has `lint_rung` + `locator_lint`), and `DECISION_PROMPT` actively *requires* naming the fact ("a condition, drug, or procedure named in the {output_kind}") — which produces world-knowledge trivia ("Which body system does a mammogram primarily evaluate?") answerable without the note. Forbid naming in the prompt and enforce it with a lint at generation time.

**Files:**
- Modify: `src/cloak/train/ladder_probes.py` (`DECISION_PV` at line 51, `DECISION_PROMPT` at lines 109–140, new `lint_decision` near `locator_lint`, `decision_probes_for_docs` at lines 631–723, `__main__` self-check)
- Modify: `scripts/build_probes.py` (both `lp.decision_probes_for_docs(...)` call sites, currently lines 265 and 506)
- Test: `src/cloak/tests/test_ladder_probe_build.py`

**Interfaces:**
- Consumes: `_tokens` (exists).
- Produces: `lint_decision(q: str, lattice_surfaces: list[str]) -> bool` (True = keep); `decision_probes_for_docs(..., lattice_surfaces_of: dict | None = None)` — new keyword arg, `{doc_id: [surface, ...]}`; `DECISION_PV = 4`.

- [ ] **Step 1: Write the failing tests**

Add to `src/cloak/tests/test_ladder_probe_build.py`:

```python
def test_lint_decision_rejects_questions_naming_a_lattice_fact():
    from cloak.train.ladder_probes import lint_decision

    surfaces = ["mammogram", "congestive heart failure"]
    # world-knowledge trivia shape: names the fact, asks a generic property of it
    assert not lint_decision("Which body system does a mammogram primarily evaluate?",
                             surfaces)
    assert not lint_decision(
        "Which specialist should manage the congestive heart failure noted in the plan?",
        surfaces)
    # circumstance-grounded question that names no fact passes
    assert lint_decision(
        "Which specialist should follow up the condition managed with daily medication?",
        surfaces)
    assert lint_decision("Which route is supported?", [])


def test_decision_probes_lint_drops_fact_naming_questions(monkeypatch, tmp_path):
    docs = [{"id": "d1", "text": "Patient has hypothyroidism, treated with Synthroid."}]
    reply = json.dumps({"decisions": [
        {"q": "Which body system does hypothyroidism affect?",
         "options": ["Endocrine", "Cardiac", "Renal"], "gold": "Endocrine",
         "depends_on": ["hypothyroidism"]},
        {"q": "Which specialist should follow up the condition managed with daily medication?",
         "options": ["Endocrinologist", "Cardiologist", "Nephrologist"],
         "gold": "Endocrinologist", "depends_on": ["hypothyroidism"]},
    ]})

    class FakeTeacher:
        def generate(self, _prompt):
            return reply

    monkeypatch.setattr(lp, "_teacher", lambda _m, _b: FakeTeacher())

    out = lp.decision_probes_for_docs(
        docs, {"d1": "CEILING NOTE"}, "clinical", workers=1, model="fake",
        cache_path=tmp_path / "decision_probes.json",
        lattice_surfaces_of={"d1": ["hypothyroidism", "Synthroid"]},
    )

    kept_qs = [e["q"] for e in out["d1"]]
    assert kept_qs == [
        "Which specialist should follow up the condition managed with daily medication?"
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py -q`
Expected: both new tests FAIL (`lint_decision` doesn't exist; `decision_probes_for_docs` rejects the unknown kwarg).

- [ ] **Step 3: Implement**

In `src/cloak/train/ladder_probes.py`:

1. Set `DECISION_PV = 4` and extend its comment block:

```python
# pv2: JSON-object output ({"decisions": [...]}) enforced via response_format=json_object.
# pv3: decisions must turn on a fact's identity/category (placeholder-breakable), not plan
#      readback or outside-knowledge appropriateness; DECISION_KINDS retargeted to category.
# pv4: the question must NOT name the target fact (world-knowledge-trivia shape measured in
#      the 2026-07-12 super sweep: every decision named its fact and asked a generic
#      property); identify the situation via documented circumstances. Enforced by
#      lint_decision at generation time.
DECISION_PV = 4
```

2. Replace `DECISION_PROMPT` with (requirement 2 is new; old 2–5 become 3–6):

```python
DECISION_PROMPT = """You design decision checks that test whether a {output_kind} still supports \
a reader's decision when the specific clinical facts in it may have been replaced by broader \
categories or hidden.

Below are a document and the {output_kind} written from it. Write up to {k} multiple-choice \
decision questions ({decision_kinds}) that a professional reading ONLY the {output_kind} must \
answer. EACH question must satisfy ALL of:
1. Its correct answer is DETERMINED BY the identity or category of ONE specific clinical fact \
(a condition, drug, or procedure in the {output_kind}) — so that if that fact were replaced by \
a generic placeholder ("a condition"), the answer could no longer be chosen, while knowing the \
fact or its category (e.g. "a cardiovascular disease") is enough to choose it. The decision \
must follow FROM what the fact clinically IS — which medical specialist should treat it, which \
body system it affects, which class of treatment or diagnosis it calls for, or how it routes a \
referral.
2. The question itself must NOT name that fact or a close synonym of it. Naming the fact and \
asking one of its generic properties is world-knowledge trivia, answerable without the \
{output_kind}. Identify the situation through the patient's OTHER documented circumstances — \
the presenting complaint, the documented course, what it is managed or treated with — so the \
question is unanswerable if the fact's meaning was lost.
3. Do NOT ask for a value written verbatim in the {output_kind} (a dose, a follow-up interval, \
the literal plan action). Those are readable no matter how the fact is anonymized and do not \
test whether the fact's meaning survived.
4. The correct answer must be SUPPORTED BY the {output_kind}'s content — pickable by a careful \
reader from what the note states, never requiring outside medical knowledge the note omits.
5. Give 3-5 options, exactly one correct, all clearly distinct and mutually exclusive; no \
yes/no questions.
6. In "depends_on", quote the exact phrase(s) NAMING the specific fact the answer depends on \
(the condition/drug/procedure), not the plan line that states the answer.

Document:
{doc}

The {output_kind}:
{out_hi}

Reply ONLY with a JSON object:
{{"decisions": [{{"q": "...", "options": ["...", "..."], "gold": "...", "depends_on": ["...", "..."]}}]}}"""
```

3. Add below `locator_lint`:

```python
def lint_decision(q, lattice_surfaces):
    """Reject decision questions that NAME a lattice fact (the world-knowledge-trivia shape:
    'Which body system does a mammogram evaluate?'). Such a question is answerable without the
    note, so it never tests whether the fact's meaning survived; the prompt (pv4) forbids it,
    this lint enforces it. ponytail: surfaces only, not profile aliases — extend to aliases if
    trivia re-slips through via synonyms."""
    qt = _tokens(q or "")
    return not any(
        st <= qt for s in lattice_surfaces or [] if (st := _tokens(s or ""))
    )
```

4. In `decision_probes_for_docs`, add the keyword arg `lattice_surfaces_of: dict | None = None` (after `cache_path`), mention it in the docstring (`lattice_surfaces_of: doc_id -> detected lattice surfaces; questions naming one are dropped by lint_decision`), and apply the lint in the kept-loop:

```python
            for row in rows[:k]:
                q, opts, gold = (
                    row.get("q", "").strip(),
                    row.get("options"),
                    row.get("gold"),
                )
                if (
                    q.endswith("?")
                    and isinstance(opts, list)
                    and 3 <= len(opts) <= 5
                    and gold in opts
                    and lint_decision(q, (lattice_surfaces_of or {}).get(d["id"], []))
                ):
```

5. In the `__main__` self-check, after the `lint_rung` asserts, add:

```python
    assert not lint_decision("Which body system does a mammogram evaluate?", ["mammogram"])
    assert lint_decision("Which specialist should follow up the screened condition?",
                         ["mammogram"])
```

In `scripts/build_probes.py`, pass the surfaces at both call sites:

6. `build_ladder` (currently line 265):

```python
        decisions = lp.decision_probes_for_docs(
            rows, out_hi_of, corpus, workers=args.workers,
            model=teacher_model, base_url=teacher_base_url,
            lattice_surfaces_of={d["id"]: [s["surface"] for s in spans_of.get(d["id"], [])]
                                 for d in rows})
```

7. `build_ladder_detected` (currently line 506): same change (its `spans_of` is already the lattice-only dict).

- [ ] **Step 4: Run tests + self-check to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_mc_reader.py -q && PYTHONPATH=src .venv/bin/python src/cloak/train/ladder_probes.py`
Expected: all PASS + self-check OK.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/ladder_probes.py scripts/build_probes.py src/cloak/tests/test_ladder_probe_build.py
git diff --cached --name-only   # must list ONLY the three files above
git commit -m "feat(qa-build): forbid fact-naming decisions (DECISION_PV 4) + lint_decision

DECISION_PROMPT required naming the fact, which produced world-knowledge trivia
answerable without the note. pv4 identifies the situation via documented
circumstances; lint_decision drops questions naming any lattice surface.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Ladder teacher asks two questions — rung 0 + rung 1 (`LADDER_PV` 4)

On nested lattices no natural-language question selects one level, so the teacher repeats one question across rungs and "4 kept rungs" is one measurement re-thresholded (measured: `hypertension`/`osteoporosis`/`kidney stones`/`arthritis` each had 1 unique question across 4–5 rungs). Decided resolution (register Issue "Per-rung ladder questions are redundant"): one exact question (rung 0) + one semantic question pinned at rung 1 (the finest generalization); acceptance = {surface, rung 1} (which `entail_score(answer, rungs, 1, aliases)` already computes); coarser fills score 0 under Task 2's binary scorer, creating the gradient toward the finest legal rung.

**Files:**
- Modify: `src/cloak/train/ladder_probes.py` (`LADDER_PV` at line 47, `LADDER_PROMPT` at lines 78–107, `ladder_probes_for_docs` prompt build at lines 540–547 and rung gate at lines 586–599)
- Test: `src/cloak/tests/test_ladder_probe_build.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LADDER_PV = 4`; cache entries unchanged in shape (`rungs` still carries the full ladder — validation and reward acceptance sets need it); only rungs 0 and 1 are ever generated/kept. `validate_ladder`, `_score_ladder`, `_with_validated_rung0` need no changes.

- [ ] **Step 1: Write the failing test**

Add to `src/cloak/tests/test_ladder_probe_build.py`:

```python
def test_ladder_generation_keeps_only_rung0_and_rung1(monkeypatch, tmp_path):
    # decided resolution for nested lattices: ONE semantic question pinned at rung 1;
    # teacher replies at coarser rungs are rejected as bad_rung, and the prompt no longer
    # shows the coarser rungs at all (no question can select among nested levels anyway)
    docs = [{"id": "d1", "text": "Patient has hypertension."}]
    span = {"surface": "hypertension", "type": "health-condition"}
    monkeypatch.setattr(
        lp, "span_levels",
        lambda s: ["artery disease", "vascular disease", "a physical condition"])

    prompts = []

    class FakeTeacher:
        def generate(self, prompt):
            prompts.append(prompt)
            return json.dumps({"probes": [
                {"rung": 0, "q": "What condition needs daily monitoring?"},
                {"rung": 1, "q": "What category of condition is being managed?"},
                {"rung": 2, "q": "What broader kind of condition is present?"},
            ]})

    monkeypatch.setattr(lp, "_teacher", lambda _m, _b: FakeTeacher())

    rejects = []
    out = lp.ladder_probes_for_docs(
        docs, {"d1": [span]}, "clinical", workers=1, model="fake",
        cache_path=tmp_path / "ladder_probes.json", reject_sink=rejects)

    assert sorted(e["rung"] for e in out["d1"]) == [0, 1]
    assert [r["gate"] for r in rejects] == ["bad_rung"]
    assert rejects[0]["rung"] == 2
    # full ladder still recorded on the entries (acceptance sets need it) ...
    assert all(e["rungs"] == ["hypertension", "artery disease", "vascular disease",
                              "a physical condition"] for e in out["d1"])
    # ... but the teacher only ever sees rung 0 and rung 1
    assert "artery disease" in prompts[0]
    assert "vascular disease" not in prompts[0]
    assert "a physical condition" not in prompts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py -q`
Expected: the new test FAILS (rung 2 currently accepted; coarser rungs currently in the prompt).

- [ ] **Step 3: Implement**

In `src/cloak/train/ladder_probes.py`:

1. Set `LADDER_PV = 4` and extend the comment:

```python
# pv2: clinical-role grounding, positional/enumeration forbidden.
# pv3: JSON-object output ({"probes": [...]}) enforced via response_format=json_object.
# pv4: two questions per span — rung 0 (exact) + rung 1 (finest generalization) — instead of
#      one per rung. On nested lattices no question selects one level (measured: identical
#      question repeated across rungs), so per-rung questions were one measurement scored at
#      L thresholds, and monotone acceptance made coarsening nearly free.
LADDER_PV = 4
```

2. Replace `LADDER_PROMPT` with (keep the existing TODO comment above it):

```python
LADDER_PROMPT = """You write probe questions used to grade how well a {output_kind} preserves \
facts at different levels of detail.

A {output_kind} will be written from the document below. The target fact may appear in it \
exactly, as the truthful generalization given below, or not at all. Write exactly TWO probe \
questions:
- one with "rung": 0, whose best answer is the exact value when the {output_kind} states the \
fact exactly;
- one with "rung": 1, asking about the PROPERTY the generalization expresses — its best answer \
is the generalization phrase, never the specific value.

EACH question must satisfy ALL of:
1. it does not contain the exact value, the generalization phrase, or close synonyms of them;
2. it identifies which fact it asks about ONLY through the fact's own clinical role — what it \
is managed or treated with, its documented status or course, or its clinical consequence. \
NEVER identify it by the fact itself; by its POSITION or ORDER in a list ("the first/last/third \
condition"); or by ENUMERATING the other facts it appears alongside ("the condition listed \
with X, Y and Z"). Position and neighbouring facts are reordered or hidden when the answer is \
graded, and a question answerable by position or by echoing a name does NOT test whether the \
fact's meaning was preserved — it must be answerable only by understanding this fact;
3. it is a wh- question (what / which / who / where / when / why / how) with a short-phrase \
answer; no yes/no questions;
4. it is answerable from the {output_kind} alone, by a reader who never saw the document.

Document:
{doc}

Target fact: "{surface}"   (type: {type}; appears in: "{sentence}")
rung 0 (exact value): {surface}
rung 1 (finest truthful generalization): {rung1}

Reply ONLY with a JSON object: {{"probes": [{{"rung": 0, "q": "..."}}, {{"rung": 1, "q": "..."}}]}}"""
```

3. In `ladder_probes_for_docs`, change the prompt build (the `todo.append` block) to pass `rung1` instead of the rung listing:

```python
                    "prompt": LADDER_PROMPT.format(
                        output_kind=kind,
                        doc=d["text"],
                        surface=s["surface"],
                        type=s.get("type", ""),
                        sentence=s.get("sent") or sentence_of(d["text"], s["surface"]),
                        rung1=rungs[1],
                    ),
```

(`rungs[1]` always exists: `want_of` only admits spans where `span_levels(s)` is non-empty, and `rung_phrases` prepends the surface. If `rungs[1]` is a semantically-empty phrase the existing `empty_gold` gate still rejects the rung-1 row post-hoc, leaving the span echo-only — same outcome as pv3.)

4. Tighten the rung gate in the reply loop from `0 <= rung < len(t["rungs"])` to:

```python
                if not (isinstance(rung, int) and rung in (0, 1)):
                    _rej(t, rung, q, "bad_rung")
                    continue
```

- [ ] **Step 4: Run tests + self-check to verify they pass**

Run: `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_ladder_probe_build.py src/cloak/tests/test_two_channel_reward.py src/cloak/tests/test_mc_reader.py -q && PYTHONPATH=src .venv/bin/python src/cloak/train/ladder_probes.py`
Expected: all PASS + self-check OK. (`test_ladder_probes_scopes_cache_to_current_spans_and_rungs` keeps passing — its fake teacher already replies at rungs 0/1 only.)

- [ ] **Step 5: Commit**

```bash
git add src/cloak/train/ladder_probes.py src/cloak/tests/test_ladder_probe_build.py
git diff --cached --name-only   # must list ONLY the two files above
git commit -m "feat(qa-build): one semantic question pinned at rung 1 (LADDER_PV 4)

On nested lattices the teacher repeated one question across rungs — L rungs were
one measurement at L thresholds, and monotone acceptance made coarsening nearly
free. pv4 asks rung 0 + rung 1 only; with binary containment scoring a coarser
fill now scores 0 on the semantic channel.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7 (controller-run, NOT a subagent task): rerun the super-teacher sweep

Regenerate the validated artifacts at pv4 so the keep/reject picture is measured, not assumed. Same scope as the validated 2026-07-12 sweep (perf-gate-passed workflow, unchanged perf profile).

- [ ] **Step 1: Pre-flight** — GPU occupancy: `rocm-smi --showpidgpus` must show no foreign GPU process; local proxy up (`curl -s localhost:8060/v1/models`); `OPENROUTER_API_KEY` present in the environment (ask the user if not).
- [ ] **Step 2: Run** (unbuffered, background, log to `results/`):

```bash
CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
  .venv/bin/python -u scripts/build_probes.py --detect --corpora clinical \
  --teacher-model "nvidia/nemotron-3-super-120b-a12b:free" \
  --teacher-base-url https://openrouter.ai/api/v1 --out-tag super
```

PV bumps invalidate the teacher caches, so all teacher calls regenerate (OpenRouter `:free` — no cost); anchors and detection replay from `CLOAK_LLM_CACHE`.
- [ ] **Step 3: Report outcomes honestly** — per-gate gen-reject counts, kept rungs per span (expect ≤ 2), decision verdict distribution (`kept`/`ceiling`/`floor`/`unlinked`), and a direct statement of whether good decisions still ceiling-reject under the MC prompt (the unconfounded evidence for the register's open (a)/(b)/(c) reader decision). Degeneracies get reported, not engineered around.
- [ ] **Step 4: Commit artifacts** (path-scoped): `data/probes_ladder_validated.super.json`, `results/ladder_generations.super.json`, `results/ladder_gen_rejects.super.json`, `results/probe_health.json`, plus the teacher caches `data/ladder_probes.json`/`data/decision_probes.json` if git-tracked.

---

## Out of scope (explicitly)

- The full teacher-prompt redesign (register Issues 3–4 directions: attribute-absent grounding, granularity-sensitive answers, reasoning/"why" decision shapes) — next session, on the unconfounded evidence from Task 7.
- The (a)/(b)/(c) stronger-reader decision — made by the user after Task 7's rerun.
- Drug/procedure lattice-profile coverage (register Issue 4a) — a data problem, not a prompt/code problem.
