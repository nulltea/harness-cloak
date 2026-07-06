# Design 3 Reconstructor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local learned reconstructor that recovers the survived-span residue the rule cascade misses — paraphrase-reworded and lossy fill mentions (D-1 semantic, D-3) — by editing `out_p` using the original surfaces in R, pushing survived-span recovery from ~82% toward the client-side ceiling (~100%).

**Architecture:** Residue-targeted constrained edit. The existing deterministic cascade (`invert()`) runs first and resolves A/B/C + alias cases; only its residue generalization entries Q are handed to a flan-t5-base + LoRA seq2seq that reads the cascade's partial output plus a linearized restore-map (fill → original, typed) and rewrites, with a **copy-bias guard** that permits only R's original surfaces as novel content (else falls back to the cascade output — do-no-harm). Training targets are distilled from the Qwen survival judge, which already grounds each reworded mention to a verbatim `out_p` quote.

**Tech Stack:** Python, PyTorch (ROCm host `.venv`), transformers 5.12 + peft 0.19 (LoRA), `google/flan-t5-base`, the `inferdpt.llm` OpenAI-compatible client (gemma out_p pin, Qwen judge pin), rapidfuzz.

## Global Constraints

- **out_p pin (unchanged):** gemma 4 (E4B), temp 0, non-thinking, `RT_BASE_URL=http://localhost:8060/v1`, `max_tokens 1024`; content-addressed cache via `INFERDPT_LLM_CACHE=data/llm_cache`. Reused verbatim from `cloak.train.roundtrip`.
- **Judge pin (target-builder):** `Qwen3.6-35B-A3B`, temp 0, non-thinking, same base_url; llama-swap serves Qwen `-np 1` (serial). Reused verbatim from `scripts/spikes/survival_by_type.py`.
- **Reconstructor model:** `google/flan-t5-base` + LoRA (peft). **Pinned to the versions verified installed in the host `.venv` this session — `transformers==5.12.1`, `peft==0.19.1`, `torch` (ROCm, CUDA-available)**; the plan is install-free (no `pip install`; never `pip install torch` — see `~/docs/torch-gpu.md`). Checkpoints under `data/models/reconstructor_v1/`.
- **One GPU process at a time.** Before any training/inference run, `pgrep -af train_pii` and this plan's own jobs; never launch a second GPU run while one is live. Always `.venv/bin/python -u`, log to `results/`.
- **Train/eval corpus disjoint (open-label generality):** train on `clinical`, evaluate held-out on `lexsum`, and vice-versa. Never report an averaged-across-corpora number; per corpus always.
- **Do-no-harm is a hard constraint:** the reconstructor may introduce ONLY original surfaces from R as novel content; it must never overwrite a span the cascade already resolved, and a false substitution (wrong surface asserted) is worse than a miss. The copy-bias guard enforces this; a violated output falls back to the cascade result.
- **Empirical honesty:** compare reconstructor vs cascade at identical upstream settings and matched realized privacy; recovery measured mention-anchored, per type; report regressions.
- **Training record:** before the training run, write `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md` (spec-then-results, schema in CLAUDE.md); fill Results after.
- `results/`, `data/` artifacts are gitignored — findings live in docs.

## File Structure

- `src/cloak/reconstruct.py` — reconstructor module: `linearize_restore_map`, `splice_at_quote`, `copy_bias_guard`, model load + `reconstruct(out_p, R)` composing cascade + seq2seq on residue. One responsibility: turn (out_p, R) → out_final via cascade+model.
- `scripts/build_reconstructor_data.py` — durable data builder: distill the Qwen judge into `(input, target)` JSONL training triples. (Durable, re-run workflow → `scripts/`, not spikes.)
- `scripts/train_reconstructor.py` — Seq2Seq + LoRA training loop (follows `scripts/train_pii_gliner.py` structure).
- `scripts/spikes/reconstructor_eval.py` — mention-anchored recovery eval vs the cascade baseline (one-off comparison → spikes).
- `tests/test_reconstruct.py` — unit tests for the pure functions (linearize, splice, guard, metric).
- `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md` — training record.

---

### Task 1: Pure helpers — linearize restore-map & splice-at-quote

**Files:**
- Create: `src/cloak/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Produces: `linearize_restore_map(residue: list[dict]) -> str` (residue entries are R dicts with `surface`, `replacement`, `type`); `splice_at_quote(text: str, quote: str, replacement: str) -> tuple[str, bool]` (case/whitespace-insensitive locate of `quote`, replace that slice with `replacement`; bool = whether a splice happened).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reconstruct.py
from cloak.reconstruct import linearize_restore_map, splice_at_quote

def test_linearize_restore_map():
    residue = [{"surface": "arthritis", "replacement": "a disease", "type": "DEM"},
               {"surface": "CABG", "replacement": "a procedure", "type": "MISC"}]
    out = linearize_restore_map(residue)
    assert out == ("DEM: \"a disease\" => \"arthritis\"\n"
                   "MISC: \"a procedure\" => \"CABG\"")

def test_splice_at_quote_case_and_space_insensitive():
    text = "Patient has Early 1980s onset."
    new, ok = splice_at_quote(text, "early  1980S", "January 13th 1982")
    assert ok and new == "Patient has January 13th 1982 onset."

def test_splice_at_quote_absent_returns_unchanged():
    new, ok = splice_at_quote("no match here", "CABG surgery", "coronary bypass")
    assert not ok and new == "no match here"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v`
Expected: FAIL with `ImportError: cannot import name 'linearize_restore_map'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cloak/reconstruct.py
"""Design 3 reconstructor: recover the survived-span residue the rule cascade
(cloak.extract.invert) misses — paraphrase-reworded / lossy fill mentions — by editing
out_p with the original surfaces in R. Residue-targeted constrained edit; copy-bias guard
enforces do-no-harm (only R originals may enter). Plan: docs/plans/2026-07-06-reconstructor-design3-plan.md.
"""
import re


def linearize_restore_map(residue: list[dict]) -> str:
    """One line per residue entry: 'TYPE: "fill" => "original"'. Order preserved."""
    return "\n".join(
        f'{e.get("type", "MISC")}: "{e["replacement"]}" => "{e["surface"]}"'
        for e in residue)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def splice_at_quote(text: str, quote: str, replacement: str) -> tuple[str, bool]:
    """Locate `quote` in `text` ignoring case and inner whitespace; replace that slice
    with `replacement`. Returns (new_text, spliced?)."""
    if not quote:
        return text, False
    pat = re.compile(r"\s+".join(re.escape(w) for w in quote.split()), re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return text, False
    return text[:m.start()] + replacement + text[m.end():], True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cloak/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: reconstructor helpers — linearize restore-map + splice-at-quote"
```

---

### Task 2: Edit-anchored guard (do-no-harm enforcement)

**Files:**
- Modify: `src/cloak/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `_norm` (Task 1).
- Produces: `edit_guard(prepass: str, candidate: str, residue: list[dict], max_edits: int) -> bool` — accept a rewrite ONLY if every diff op (vs `prepass`) is a REPLACE whose deleted span is non-empty and whose inserted text carries a residue entry's original surface, with ≤ `max_edits` regions. Anchored, not substring-of-allowed (Round-2 fix): rejects pure insertions (surface dropped at the wrong place), pure deletions (mention removed without restoring), and inserts that don't carry a residue original.

**Why anchored** (reviewer): a word-set guard passes `"B filed against A"`↔`"A filed against B"` (reorder); a substring-of-allowed guard still passes "insert the surface elsewhere while deleting the quote". Requiring every edit to be an in-place REPLACE that puts a residue original where a mention was closes both.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_reconstruct.py
from cloak.reconstruct import edit_guard

def _res(*pairs): return [{"surface": s, "replacement": r} for s, r in pairs]

def test_guard_accepts_in_place_restore():
    prepass = "Patient has a disease and takes a drug."
    cand = "Patient has arthritis and takes a drug."   # replace 'a disease' -> 'arthritis'
    assert edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_hallucinated_insert():
    prepass = "Patient has a disease."
    cand = "Patient has arthritis in Boston."           # inserted 'arthritis in Boston' not tight
    assert not edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_insert_elsewhere_with_deletion():
    prepass = "Filed in the early 1980s."
    cand = "January 13th 1982 note. Filed recently."    # pure insert up front + quote deleted
    assert not edit_guard(prepass, cand,
                          _res(("January 13th 1982", "the early 1980s")), max_edits=3)

def test_guard_rejects_wrong_location_replace():
    prepass = "The org filed. Patient has a disease."
    cand = "arthritis filed. Patient has a disease."    # surface put where 'The org' was, not at 'a disease'
    assert not edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_too_many_edits():
    prepass = "a disease and a drug."
    cand = "arthritis and lasix."                        # 2 valid in-place edits > max_edits=1
    assert not edit_guard(prepass, cand,
                          _res(("arthritis", "a disease"), ("lasix", "a drug")), max_edits=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_guard_rejects_reorder -v`
Expected: FAIL with `ImportError` on `edit_guard`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cloak/reconstruct.py  (add `import difflib` at top)
def edit_guard(prepass: str, candidate: str, residue: list[dict], max_edits: int) -> bool:
    """do-no-harm, fill-occurrence anchored (Round-2/3 fix). At inference there is no judge
    quote, so anchor to each residue entry's own mention: fuzzy-locate the entry's fill in
    `prepass`, and require EVERY diff op to be a REPLACE that (a) overlaps one entry's located
    fill span, (b) inserts that SAME entry's surface (tight match, not just any allowed
    surface, and not surrounded by extra content), (c) has a non-empty deleted span. Rejects
    pure insert (surface dropped elsewhere), pure delete (mention removed w/o restoring),
    wrong-location replace, multi-surface hunks, and >max_edits regions.
    ponytail: char-diff + fuzzy-anchor guard with a known ceiling — upgrade to token-
    constrained decoding (prefix_allowed_tokens_fn) if the measured reject/fallback rate is
    high."""
    from rapidfuzz import fuzz
    anchors = []   # (lo, hi, surface_norm) — fill mention span in prepass + its target surface
    for e in residue:
        al = fuzz.partial_ratio_alignment(e["replacement"].lower(), prepass.lower())
        if al and al.score >= 60.0 and al.dest_end > al.dest_start:
            anchors.append((al.dest_start, al.dest_end, _norm(e["surface"])))
    # Round-4 hardening: reject when two anchors of DIFFERENT surfaces overlap the same region
    # (ambiguous fuzzy match on a repeated/generic phrase — we cannot tell which original the
    # edit restores). Bail rather than risk a wrong-surface substitution.
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            (lo1, hi1, s1), (lo2, hi2, s2) = anchors[a], anchors[b]
            if s1 != s2 and not (hi1 <= lo2 or hi2 <= lo1):
                return False
    edits = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=prepass, b=candidate, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        edits += 1
        deleted, inserted = _norm(prepass[i1:i2]), _norm(candidate[j1:j2])
        if not deleted or not inserted:
            return False                      # not an in-place replace
        matched = False
        for lo, hi, surf in anchors:
            overlaps = not (i2 <= lo or i1 >= hi)   # edit region meets this fill's mention
            tight = surf and (inserted == surf or        # exact, or surface + minor punctuation/casing
                              (surf in inserted and len(inserted) - len(surf) <= 3))
            if overlaps and tight:
                matched = True
                break
        if not matched:
            return False                      # wrong location, wrong/mixed surface, or hallucination
    return edits <= max_edits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cloak/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: reconstructor edit-anchored guard (do-no-harm, difflib)"
```

---

### Task 3: Training-data builder — distill the survival judge

**Files:**
- Create: `scripts/build_reconstructor_data.py`
- Test: `tests/test_reconstruct.py` (target-construction unit only)

**Interfaces:**
- Consumes: `linearize_restore_map`, `splice_at_quote`, `build_target`, `restorable` (Task 1 + below); `cloak.extract._rule_prepass` (returns `(text, stats, residue)`); the judge machinery from `scripts/spikes/survival_by_type.py` (`build_jobs`, `_judge`, `parse_judge`, `JUDGE_TMPL`, `SYSTEM`, `grounded`, `fill_present`, `exact_present`); the type-sanity check `cloak.extract._type_sane(entity_type, fill, window)` (already in extract.py); `cloak.train.roundtrip.roundtrip_batch`.
- Produces: JSONL at `data/reconstructor_<corpus>.jsonl`, one row `{"input": str, "target": str, "corpus": str, "doc_id": str, "n_residue": int, "n_edits": int, "is_noop": bool}`. `input` = `<cascade out_final'>\n\n[RESTORE]\n<linearized residue>`; `target` = `input`'s text region with each **admitted** located mention spliced to its original, unchanged elsewhere. Two failure modes are deliberately trained as **no-op targets** (`is_noop=True` when `n_edits==0`): (a) judge abstains (ABSENT/TEMPLATED), (b) the admission gate rejects the correspondence. A separate `data/reconstructor_<corpus>_degeneracies.jsonl` logs every rejected-but-grounded case for audit.

**Target-admission gate** (Round-1 #1 + Round-2 refinement — a grounded quote is NOT sufficient, and type-sanity alone is NOT correspondence: "the last four years" vs "three years ago" are both DATETIME-sane, yet splicing the original into that D-4 case teaches the false-restoration the project calls worse than a miss). A located mention is admitted only when ALL hold:
1. judge `label ∈ {SURVIVED, REWORDED}` and `grounded(quote, prepass)`;
2. `_type_sane(entry.type, entry.replacement, quote)` — mention is the span's TAB type;
3. **mandatory correspondence for scalar/named ambiguous types** (`DATETIME, QUANTITY, LOC, ORG`): an NLI check (`cross-encoder/nli-deberta-v3-small`) that the quote is entailed by / no more specific than the FILL (`quote ⊨ fill`, `fill ⊬ contradiction`) — so the mention is the fill's restatement, not a model-invented specific. Fail-closed: an ambiguous type with no NLI checker abstains. D-3 lossy (quote consistent with fill) passes; D-4 (quote asserts a specific absent from the fill) is rejected.
Rejected-but-grounded cases become no-op targets and are logged to the degeneracies file. **Pre-training audit:** hand-check 30 admitted targets, report admitted-target precision in the training record; the run does not proceed at < ~0.95 precision.

`restorable(entry, verdict, prepass, nli) -> bool` implements the gate; `build_target` (Task 1) applies it.

- [ ] **Step 1: Write the failing test** (target construction is the only unit-testable piece)

```python
# add to tests/test_reconstruct.py
def test_build_target_splices_located_mentions_only():
    from cloak.reconstruct import build_target
    text = "The org filed in Early 1980s."
    # judge verdicts: one located (REWORDED), one abstain (ABSENT quote=None)
    located = [{"surface": "January 13th 1982", "quote": "Early 1980s"},
               {"surface": "Hamilton County Court", "quote": None}]
    tgt, n = build_target(text, located)
    assert tgt == "The org filed in January 13th 1982." and n == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_build_target_splices_located_mentions_only -v`
Expected: FAIL with `ImportError: cannot import name 'build_target'`

- [ ] **Step 3: Implement `build_target` in the module, then the builder script**

```python
# add to src/cloak/reconstruct.py
def build_target(text: str, located: list[dict]) -> tuple[str, int]:
    """Splice each located mention's original into text (longest-quote-first so a short quote
    can't match inside a longer one); entries with a falsy quote are abstains (no edit).
    Returns (target_text, n_edits)."""
    edits = 0
    for e in sorted([x for x in located if x.get("quote")],
                    key=lambda x: -len(x["quote"])):
        new, ok = splice_at_quote(text, e["quote"], e["surface"])
        if ok:
            text, edits = new, edits + 1
    return text, edits


_AMBIGUOUS_TYPES = {"DATETIME", "QUANTITY", "LOC", "ORG"}  # scalar/named — high false-corr risk


def restorable(entry: dict, verdict: dict, prepass: str, nli=None) -> bool:
    """Target-admission gate (Round-2 fix — type-sanity alone is NOT correspondence: "the
    last four years" vs "three years ago" are both DATETIME-sane). Admit a splice target
    only if the judge marked it present, the quote is grounded in the text we edit, AND:
      - for scalar/named ambiguous types, a MANDATORY correspondence check: the quote must
        be entailed by / consistent with the FILL (the generalization actually sent), so the
        mention is the fill's restatement, not a different specific the model invented. This
        admits D-3 lossy (quote ⊨ fill: "Early 1980s" is consistent with fill "the early
        1980s") and rejects D-4 ("three years ago" is NOT entailed by fill "some time ago"'s
        content — it adds a specific not in doc_p).
      - other types: type-sanity suffices.
    D-3 (exact original in R) still passes; only D-4 false-correspondence is filtered."""
    from cloak.extract import _type_sane
    q = verdict.get("quote")
    if verdict.get("label") not in ("SURVIVED", "REWORDED") or not q:
        return False
    if _norm(q) not in _norm(prepass):
        return False
    if not _type_sane(entry.get("type", "MISC"), entry["replacement"], q):
        return False
    if entry.get("type", "MISC") in _AMBIGUOUS_TYPES:
        # mandatory correspondence: quote must not assert a specific beyond the fill's content
        return _corresponds(entry["replacement"], q, nli)   # NLI: quote ⊨ fill, no added specificity
    return True


def _corresponds(fill: str, quote: str, nli) -> bool:
    """Admit only if the quote adds NO information beyond the generalized fill — i.e.
    `fill ⊨ quote` (Round-3 fix: the entailment must run fill→quote, not quote→fill; the
    latter admits a model-invented specific because "three years ago" ⊨ "some time ago").
    Admits D-3 lossy ("the early 1980s" ⊨ "Early 1980s"); rejects D-4 ("some time ago" ⊭
    "three years ago") and inference leaks ("a city" ⊭ "Boston"). For DATETIME/QUANTITY,
    prefer the deterministic `_value_compatible` check first (NLI is unreliable on scalars);
    fall back to NLI only when it abstains."""
    ok = _value_compatible(fill, quote)
    if ok is not None:
        return ok
    if nli is None:
        return False                       # fail-closed
    return nli(premise=fill, hypothesis=quote) == "entailment"


def _value_compatible(fill: str, quote: str):
    """Deterministic scalar gate, FAIL-CLOSED (Round-4 fix — never admit on a bare digit
    subset: "early 1980s"→"late 1980s" shares {1980} but flips the modifier). Only two
    non-deferring outcomes:
      False — the quote introduces a digit-run the fill lacks (model-invented specific/leak);
      None  — digits are subset-compatible but NOT proven equivalent, OR no digits → NLI (or
              a real date/quantity normalizer) must still approve.
    It never returns True; equivalence is proven downstream, not assumed here."""
    fn = set(re.findall(r"\d+", fill))
    qn = set(re.findall(r"\d+", quote))
    if qn and not qn <= fn:
        return False       # hard reject: a specific number absent from the fill
    return None            # defer: subset-compatible or no digits — NLI must confirm fill ⊨ quote


def _load_nli():
    """cross-encoder/nli-deberta-v3-small -> callable(premise, hypothesis) ->
    {'entailment'|'neutral'|'contradiction'}. Local, small; loaded once."""
    from transformers import pipeline
    pipe = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
    label_map = {"entailment": "entailment", "neutral": "neutral",
                 "contradiction": "contradiction"}
    def _nli(premise: str, hypothesis: str) -> str:
        out = pipe({"text": premise, "text_pair": hypothesis})
        return label_map.get(out["label"].lower(), "neutral")
    return _nli
```

```python
# scripts/build_reconstructor_data.py
"""Distill the Qwen survival judge into reconstructor training triples.

Per doc: run the rule cascade to get out_final' + residue Q; ask the judge to locate each
residue fill's (possibly reworded) mention in out_p; build the gold target by splicing the
original at each grounded quote. Abstains (D-4/absent) keep the text unchanged, teaching
the model when NOT to edit. Emits data/reconstructor_<corpus>.jsonl.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/build_reconstructor_data.py \
       --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
       --corpora clinical --n-docs 80 --workers 6
"""
import argparse, json
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, SYSTEM, JUDGE_TMPL)
from cloak.extract import _rule_prepass
from cloak.reconstruct import linearize_restore_map, build_target, restorable, _load_nli
from cloak.train.roundtrip import roundtrip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", required=True); ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge()
    nli = _load_nli()   # cloak.reconstruct helper wrapping cross-encoder/nli-deberta-v3-small
    per_corpus: dict[str, list[dict]] = {}
    degens: dict[str, list[dict]] = {}

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)  # cascade output
        if not residue:
            continue
        items = "\n".join(f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type","MISC")}]'
                          for i, e in enumerate(residue))
        verdicts = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                              system=SYSTEM), len(residue))
        located, degen = [], []
        for e, v in zip(residue, verdicts):
            # ADMISSION GATE: admit as a splice target only if grounded in the text we edit
            # AND type-consistent correspondence holds (rejects D-4 false-correspondence).
            if restorable(e, v, prepass, nli=nli):
                located.append({"surface": e["surface"], "quote": v.get("quote")})
            else:
                located.append({"surface": e["surface"], "quote": None})  # -> no-op
                if v.get("quote") and v.get("label") in ("SURVIVED", "REWORDED"):
                    degen.append({"doc_id": m["doc_id"], "surface": e["surface"],
                                  "fill": e["replacement"], "type": e.get("type", "MISC"),
                                  "quote": v["quote"], "label": v["label"],
                                  "reason": "grounded but failed type-consistency (D-4-like)"})
        inp = f"{prepass}\n\n[RESTORE]\n{linearize_restore_map(residue)}"
        target, n_edits = build_target(prepass, located)
        per_corpus.setdefault(m["corpus"], []).append(
            {"input": inp, "target": target, "corpus": m["corpus"], "doc_id": m["doc_id"],
             "n_residue": len(residue), "n_edits": n_edits, "is_noop": n_edits == 0,
             "high_risk_noop": n_edits == 0 and len(degen) > 0})  # doc had a D-4-like reject
        degens.setdefault(m["corpus"], []).extend(degen)

    for corpus, rows in per_corpus.items():
        Path(f"data/reconstructor_{corpus}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        Path(f"data/reconstructor_{corpus}_degeneracies.jsonl").write_text(
            "\n".join(json.dumps(r) for r in degens.get(corpus, [])))
        edits = sum(r["n_edits"] for r in rows)
        noops = sum(r["is_noop"] for r in rows)
        print(f"{corpus}: {len(rows)} docs w/ residue | {edits} admitted edits | "
              f"{noops} no-op targets | {len(degens.get(corpus, []))} logged degeneracies")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then build the data**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v` → PASS (9 tests)
Then (GPU/proxy; check `pgrep -af train_pii` first):
```bash
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes .venv/bin/python -u \
  scripts/build_reconstructor_data.py --env data/ranker_env_pilot.json \
  --arms data/task_arms_pilot.json --corpora clinical,lexsum --n-docs 80 --workers 6 \
  > results/build_reconstructor_data.log 2>&1
```
Expected: `data/reconstructor_clinical.jsonl` + `data/reconstructor_lexsum.jsonl`; log prints located-edit counts per corpus (sanity: edits > 0, on the order of the D-1/D-3 counts from the survival run).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_reconstructor_data.py src/cloak/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: reconstructor training-data builder (distill survival judge)"
```

---

### Task 4: Training script — flan-t5-base + LoRA

**Files:**
- Create: `scripts/train_reconstructor.py`
- Create: `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md` (spec sections filled before the run)

**Interfaces:**
- Consumes: `data/reconstructor_<corpus>.jsonl` (Task 3).
- Produces: LoRA checkpoint `data/models/reconstructor_v1/` loadable by Task 5.

- [ ] **Step 1: Write the training record spec** (before the run, per CLAUDE.md schema)

Create `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md` with frontmatter (`type: training-experiment`, `status: planned`, `created: 2026-07-06`, `model: google/flan-t5-base`, `dataset: reconstructor_clinical.jsonl (train) — residue-edit triples distilled from Qwen judge`, `result: pending`, `companion: docs/plans/2026-07-05-survived-recovery-extractor.md`) and the required sections (Objective, Training data + ratios, Config, Selection/operating point, Evaluation & success criteria, Results=pending, Ablations, Cost, Risks, Artifacts, Sources). Objective: recover D-1/D-3 residue; success = mention-anchored recovery on held-out lexsum beats the cascade at zero false substitutions (Task 5).

- [ ] **Step 2: Write the training script**

```python
# scripts/train_reconstructor.py
"""Fine-tune flan-t5-base + LoRA on residue-edit triples (scripts/build_reconstructor_data.py).
Follows scripts/train_pii_gliner.py's HF-Trainer+PEFT pattern. Local ROCm GPU, one process.

Run: PYTHONPATH=src .venv/bin/python -u scripts/train_reconstructor.py \
       --train data/reconstructor_clinical.jsonl --out data/models/reconstructor_v1 \
       --epochs 4 --bs 4 > results/train_reconstructor.log 2>&1
"""
import argparse, json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)

BASE = "google/flan-t5-base"
PROMPT = ("Restore the original terms in the CLINICAL/LEGAL answer below. Replace each "
          "generalized mention with its original from the RESTORE map; copy everything else "
          "verbatim; if a mapped term is not present, leave the text unchanged.\n\n{input}")


def load(path, tok, noop_cap=0.3):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    # Keep all residue-positive edits. For no-ops (Round-1 #5 / Round-2 #5): abstention on
    # high-risk rejects (D-4-like / scalar/date) is a SAFETY signal, not class-balance noise
    # — keep ALL of those uncapped; cap only generic no-ops at noop_cap of the total.
    pos = [r for r in rows if not r.get("is_noop")]
    risky = [r for r in rows if r.get("is_noop") and r.get("high_risk_noop")]
    generic = [r for r in rows if r.get("is_noop") and not r.get("high_risk_noop")]
    keep_generic = generic[:int(noop_cap / (1 - noop_cap) * len(pos))] if pos else generic[:len(generic)//3]
    rows = pos + risky + keep_generic
    print(f"train rows: {len(pos)} positive + {len(risky)} high-risk no-op (kept all) + "
          f"{len(keep_generic)} generic no-op (of {len(generic)})")
    def enc(r):
        x = tok(PROMPT.format(input=r["input"]), truncation=True, max_length=1024)
        y = tok(text_target=r["target"], truncation=True, max_length=1024)
        x["labels"] = y["input_ids"]
        return x
    return Dataset.from_list(rows).map(enc, remove_columns=["input", "target", "corpus",
                                       "doc_id", "n_residue", "n_edits", "is_noop",
                                       "high_risk_noop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()
    ds = load(args.train, tok)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs, learning_rate=2e-4, bf16=True,
            logging_steps=10, save_strategy="epoch", report_to=[]),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()
    trainer.save_model(args.out); tok.save_pretrained(args.out)
    print(f"saved LoRA reconstructor -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Perf-gate the run, then train** (check `pgrep -af train_pii`; wall estimate: base+LoRA on a few hundred short triples, a few min/epoch — well under the 10-min gate. If the corpus is large, confirm batch saturation first.)

```bash
PYTHONPATH=src .venv/bin/python -u scripts/train_reconstructor.py \
  --train data/reconstructor_clinical.jsonl --out data/models/reconstructor_v1 \
  --epochs 4 --bs 4 > results/train_reconstructor.log 2>&1
```
Expected: `results/train_reconstructor.log` shows decreasing loss and `saved LoRA reconstructor -> data/models/reconstructor_v1`.

- [ ] **Step 4: Smoke-check the checkpoint loads and edits one example**

Run:
```bash
PYTHONPATH=src .venv/bin/python -u -c "
from cloak.reconstruct import load_reconstructor, run_model
m = load_reconstructor('data/models/reconstructor_v1')
print(run_model(m, 'The org filed in Early 1980s.\n\n[RESTORE]\nDATETIME: \"early 1980s\" => \"January 13th 1982\"'))
"
```
Expected: output contains "January 13th 1982" (Task 5 adds `load_reconstructor`/`run_model`; if running Task 4 first, defer this substep to after Task 5).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_reconstructor.py research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md
git commit -m "feat: reconstructor training script + FT-reconstructor v1 record (spec)"
```

---

### Task 5: Inference wrapper — cascade + model on residue, with guard

**Files:**
- Modify: `src/cloak/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `_rule_prepass` (`cloak.extract`), `linearize_restore_map`, `edit_guard`, `_finalize` (`cloak.extract`).
- Produces: `load_reconstructor(path) -> obj`; `run_model(obj, prompt: str) -> str`; `reconstruct(out_p: str, R: list[dict], model=None) -> tuple[str, dict]` — cascade first; if residue and a model is given, run the model on `<prepass>\n\n[RESTORE]\n<map>`, accept via `edit_guard(prepass, cand, residue, max_edits=2*len(residue)+1)` (else keep cascade output), then `_finalize`. Signature-compatible with `invert()` so it drops into any caller.

- [ ] **Step 1: Write the failing test** (guard fallback is the testable contract; model stubbed)

```python
# add to tests/test_reconstruct.py
from cloak.reconstruct import reconstruct

class _StubModel:
    def __init__(self, reply): self.reply = reply
    def __call__(self, prompt): return self.reply

def test_reconstruct_accepts_guarded_edit(monkeypatch):
    import cloak.reconstruct as rc
    monkeypatch.setattr(rc, "run_model", lambda m, p: m(p))
    R = [{"action": "generalize", "surface": "arthritis", "replacement": "a disease", "type": "DEM"}]
    out_p = "Patient has a disease."
    # model correctly restores; only 'arthritis' (an allowed surface) is novel -> accepted
    text, stats = reconstruct(out_p, R, model=_StubModel("Patient has arthritis."))
    assert "arthritis" in text

def test_reconstruct_rejects_hallucinated_edit_falls_back(monkeypatch):
    import cloak.reconstruct as rc
    monkeypatch.setattr(rc, "run_model", lambda m, p: m(p))
    R = [{"action": "generalize", "surface": "arthritis", "replacement": "a disease", "type": "DEM"}]
    out_p = "Patient has a disease."
    # model hallucinates 'Boston' -> edit_guard rejects -> cascade output kept (still 'a disease')
    text, stats = reconstruct(out_p, R, model=_StubModel("Patient has arthritis in Boston."))
    assert "Boston" not in text and stats.get("gen_recon_rejected") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_reconstruct_rejects_hallucinated_edit_falls_back -v`
Expected: FAIL with `ImportError`/`AttributeError` on `reconstruct`

- [ ] **Step 3: Implement**

```python
# add to src/cloak/reconstruct.py
from cloak.extract import _rule_prepass, _finalize

_RECON = {}


def load_reconstructor(path: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    if path not in _RECON:
        tok = AutoTokenizer.from_pretrained(path)
        base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",
                                                     torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).eval()
        _RECON[path] = (model, tok)
    return _RECON[path]


def run_model(obj, prompt: str) -> str:
    import torch
    model, tok = obj
    PROMPT = ("Restore the original terms in the CLINICAL/LEGAL answer below. Replace each "
              "generalized mention with its original from the RESTORE map; copy everything "
              "else verbatim; if a mapped term is not present, leave the text unchanged.\n\n{input}")
    ids = tok(PROMPT.format(input=prompt), return_tensors="pt", truncation=True,
              max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1024, num_beams=1)
    return tok.decode(out[0], skip_special_tokens=True)


def reconstruct(out_p: str, R: list[dict], model=None) -> tuple[str, dict]:
    """Cascade first; model edits only the residue, gated by copy-bias (do-no-harm)."""
    prepass, stats, residue = _rule_prepass(out_p, R, semantic=True)
    text = prepass
    if residue and model is not None:
        prompt = f"{prepass}\n\n[RESTORE]\n{linearize_restore_map(residue)}"
        cand = run_model(model, prompt)
        if edit_guard(prepass, cand, residue, max_edits=2 * len(residue) + 1):
            text = cand
            stats["gen_reconstructor"] = sum(1 for e in residue if e["surface"] in cand
                                             and e["surface"] not in prepass)
        else:
            stats["gen_recon_rejected"] = 1
    stats.setdefault("gen_reconstructor", 0)
    stats["gen_absent"] += len(residue)  # cascade's residue accounting; refined by recon count
    return _finalize(text, stats)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cloak/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: reconstruct() — cascade + guarded model edit on residue"
```

---

### Task 6: Mention-anchored recovery eval vs cascade

**Files:**
- Create: `scripts/spikes/reconstructor_eval.py`
- Test: `tests/test_reconstruct.py` (metric unit)

**Interfaces:**
- Consumes: the survival judge machinery (`survival_by_type`), `cloak.extract.invert` (baseline), `cloak.reconstruct.reconstruct` + `load_reconstructor` (Task 5), `roundtrip_batch`.
- Produces: `results/reconstructor_eval.json` — **per-residue quote-anchored** classification of each survived span under cascade vs cascade+reconstructor, aggregated per (strata × type). Metric helper `classify_recovery(out_final, quote, surface) -> str` returning one of `recovered` / `wrong_insert` / `deletion` / `miss` (Round-1 weakness #3: a bare "surface present ∧ quote gone" is gameable — a model can insert the surface elsewhere or delete the quote without restoring).

**Evaluation design** (Round-1 weakness #4 — clinical→lexsum alone tests transfer, not recovery). Run three strata, reported separately, never averaged: (a) **clinical held-out** (train on a clinical doc-split, eval the held-out clinical split), (b) **lexsum held-out** (symmetric), (c) **cross-domain** (train clinical, eval lexsum). Within each, stratify counts by D-class and TAB type. A win must hold on in-domain held-out, not only cross-domain.

**No-harm gate** (weakness #3): separately measure, over spans the cascade ALREADY resolved (A/B/C), how many the reconstructor changed. `harm_rate` must be 0 — any already-correct span the model altered is a regression and a hard fail regardless of D-class gains.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_reconstruct.py
def test_classify_recovery():
    from cloak.reconstruct import classify_recovery
    prepass = "filed in Early 1980s today"
    # window now holds the original, quote gone -> recovered
    assert classify_recovery("filed in January 13th 1982 today", "Early 1980s",
                             "January 13th 1982", prepass) == "recovered"
    # nothing changed at the mention -> miss
    assert classify_recovery("filed in Early 1980s today", "Early 1980s",
                             "January 13th 1982", prepass) == "miss"
    # quote gone but no surface in-window -> deletion (reworded away, not restored)
    assert classify_recovery("filed in today", "Early 1980s",
                             "January 13th 1982", prepass) == "deletion"
    # surface inserted ELSEWHERE, mention untouched -> miss at the mention (anti-gaming: the
    # window-local metric refuses to credit a wrong-location insert as recovery)
    assert classify_recovery("January 13th 1982 note. filed in Early 1980s today",
                             "Early 1980s", "January 13th 1982", prepass) == "miss"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_classify_recovery -v`
Expected: FAIL with `ImportError` on `classify_recovery`

- [ ] **Step 3: Implement metric + eval script**

```python
# add to src/cloak/reconstruct.py
def classify_recovery(out_final: str, quote: str, surface: str, prepass: str) -> str:
    """Per-residue outcome, evaluated in the quote's LOCAL WINDOW (Round-3 fix — a global
    'surface present ∧ quote gone' counts a wrong-location insert as recovery). Anchor on the
    words flanking the quote in `prepass`, relocate that window in `out_final`, and judge only
    there:
      recovered    — window now holds the original surface, quote no longer stands
      wrong_insert — surface appears but the quote still stands in-window
      deletion     — quote gone but surface absent in-window (reworded away, not restored)
      miss         — quote still stands, surface absent
    """
    p, f, sn, ql = _norm(prepass), _norm(out_final), _norm(surface), _norm(quote or "")
    i = p.find(ql) if ql else -1
    if i < 0:                                   # can't locate the quote — fall back to global
        window = f
    else:
        left = " ".join(p[max(0, i - 24):i].split()[-3:])
        right = " ".join(p[i + len(ql):i + len(ql) + 24].split()[:3])
        lo = (f.find(left) + len(left)) if left and left in f else 0
        hi = f.find(right, lo) if right and right in f else -1
        window = f[lo:hi] if hi > lo else f[lo:lo + max(len(sn), len(ql)) + 40]
    has_surf = sn in window
    quote_stands = bool(ql) and ql in window and sn not in ql
    if has_surf and not quote_stands:
        return "recovered"
    if has_surf and quote_stands:
        return "wrong_insert"
    if not has_surf and not quote_stands:
        return "deletion"
    return "miss"
```

```python
# scripts/spikes/reconstructor_eval.py
"""Quote-anchored recovery of survived spans: cascade (invert) vs cascade+reconstructor.
Per-residue outcomes (recovered/wrong_insert/deletion/miss) + a no-harm rate over spans the
cascade already resolved. Run ONCE PER STRATUM (Round-1 weakness #4), never averaged:
  clinical held-out:  --corpora clinical --doc-split heldout   (--ckpt trained on clinical split)
  lexsum held-out:    --corpora lexsum   --doc-split heldout   (--ckpt trained on lexsum split)
  cross-domain:       --corpora lexsum   --doc-split all       (--ckpt trained on clinical)
--doc-split heldout keeps only doc_ids not in the training split file (data/recon_train_ids_<corpus>.txt,
written by the data builder / a 80-20 hash split); 'all' uses every doc.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/spikes/reconstructor_eval.py \
       --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
       --corpora lexsum --n-docs 80 --doc-split all --ckpt data/models/reconstructor_v1
"""
import argparse, json
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, exact_present,
                              fill_present, SYSTEM, JUDGE_TMPL)
from cloak.extract import invert, _rule_prepass
from cloak.reconstruct import reconstruct, load_reconstructor, classify_recovery
from cloak.train.roundtrip import roundtrip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", required=True); ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--doc-split", choices=["all", "heldout"], default="all")
    args = ap.parse_args()

    train_ids = set()
    if args.doc_split == "heldout":
        for c in args.corpora.split(","):
            p = Path(f"data/recon_train_ids_{c}.txt")
            if p.exists():
                train_ids |= set(p.read_text().split())

    jobs, metas = build_jobs(args)
    keep = [i for i, m in enumerate(metas) if m["doc_id"] not in train_ids]
    jobs, metas = [jobs[i] for i in keep], [metas[i] for i in keep]
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge(); model = load_reconstructor(args.ckpt)
    OUT = ("recovered", "wrong_insert", "deletion", "miss")
    rows = {}   # type -> {survived, cascade:{outcome:n}, recon:{outcome:n}}
    harm = {"resolved": 0, "changed_by_recon": 0}

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        gens = [e for e in m["R"] if e["action"] == "generalize"]
        if not gens: continue
        items = "\n".join(f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type","MISC")}]'
                          for i, e in enumerate(gens))
        v = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                       system=SYSTEM), len(gens))
        casc = invert(out_p, m["R"])[0]
        recon = reconstruct(out_p, m["R"], model=model)[0]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)
        residue_surf = {e["surface"] for e in residue}
        for e, vv in zip(gens, v):
            q = vv.get("quote"); lbl = vv.get("label", "ABSENT")
            surv = exact_present(e["replacement"], out_p) or (
                lbl in ("SURVIVED", "REWORDED") and grounded(q, out_p))
            if not surv or (not fill_present(e["replacement"], out_p)
                            and exact_present(e["surface"], out_p)):
                continue   # not substituted-content survival
            # no-harm: spans the cascade already resolved (not in residue) must be untouched
            if e["surface"] not in residue_surf:
                harm["resolved"] += 1
                if casc.count(e["surface"]) != recon.count(e["surface"]):
                    harm["changed_by_recon"] += 1
                continue
            t = e.get("type", "MISC")
            r = rows.setdefault(t, dict(survived=0, cascade={k: 0 for k in OUT},
                                        recon={k: 0 for k in OUT}))
            r["survived"] += 1
            r["cascade"][classify_recovery(casc, q, e["surface"], prepass)] += 1
            r["recon"][classify_recovery(recon, q, e["surface"], prepass)] += 1

    report = {"ckpt": args.ckpt, "corpora": args.corpora, "doc_split": args.doc_split,
              "rows": rows, "harm": harm,
              "harm_rate": round(harm["changed_by_recon"] / harm["resolved"], 4)
                           if harm["resolved"] else None,
              "totals": {arm: {k: sum(r[arm][k] for r in rows.values()) for k in OUT}
                         for arm in ("cascade", "recon")}}
    Path("results/reconstructor_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test, then the eval (all three strata)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v` → PASS (12 tests)
Then, once per stratum (check `pgrep -af train_pii` first; each writes/overwrites `results/reconstructor_eval.json`, so copy each aside):
```bash
# cross-domain (train clinical, eval all lexsum)
... reconstructor_eval.py --corpora lexsum --doc-split all --ckpt data/models/reconstructor_v1 ...
  && cp results/reconstructor_eval.json results/reconstructor_eval_crossdomain.json
# clinical held-out (ckpt trained on the clinical train split)
... reconstructor_eval.py --corpora clinical --doc-split heldout --ckpt data/models/reconstructor_clinical_split ...
  && cp results/reconstructor_eval.json results/reconstructor_eval_clinical_heldout.json
# lexsum held-out
... reconstructor_eval.py --corpora lexsum --doc-split heldout --ckpt data/models/reconstructor_lexsum_split ...
  && cp results/reconstructor_eval.json results/reconstructor_eval_lexsum_heldout.json
```
Success criteria (all must hold, reported per stratum and never averaged):
- `recon.recovered > cascade.recovered` on the D-residue types, per stratum incl. at least one in-domain held-out (not only cross-domain);
- `recon.wrong_insert == 0` and `recon.deletion` not worse than cascade — a wrong_insert or a new deletion is a false/at-mention failure;
- `harm_rate == 0` — the reconstructor changed zero cascade-resolved spans. Any nonzero `harm_rate` or `wrong_insert` is a hard fail → tighten `edit_guard` bound / admission gate / no-op weighting before claiming a win.

- [ ] **Step 5: Fill the training record Results & commit**

Set `status: done` in `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md`; fill Results with the measured per-type recovery (report the win AND any regression/false-sub, per empirical-honesty). Cross-link the survived-recovery design doc and the experiment record.

```bash
git add scripts/spikes/reconstructor_eval.py src/cloak/reconstruct.py tests/test_reconstruct.py \
        research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md
git commit -m "feat: reconstructor eval (mention-anchored recovery vs cascade) + v1 results"
```

---

## Self-Review

**Spec coverage:** residue-targeted edit (Task 5 `reconstruct`), judge-distilled targets with an admission gate (Task 3 `restorable` + degeneracy log), edit-anchored do-no-harm (Task 2 `edit_guard` + Task 5 fallback), abstain on D-4/absent trained as no-op targets (Task 3), quote-anchored per-stratum eval + no-harm gate (Task 6), training record spec-then-results (Tasks 4/6). D-2 acronym/alias is handled by the deterministic proposers in the companion survived-recovery design, not this model — noted so it is not a gap.

**Round-1 reviewer fixes applied:** (1) target-admission gate (`restorable`, type-consistent correspondence) so grounded-but-wrong D-4 quotes become no-op targets, logged; (2) `edit_guard` (difflib edit-anchored, numbers included, bounded edits) replaces the word-set guard; (3) `classify_recovery` per-residue outcomes + `harm_rate` no-harm gate replace the gameable single boolean; (4) three eval strata (clinical held-out / lexsum held-out / cross-domain), never averaged; (5) no-op cap in training (`load(noop_cap=0.3)`) + pre-training admitted-target audit; (6) versions pinned to verified-installed (`transformers==5.12.1`, `peft==0.19.1`).

**Placeholder scan:** none — every code step is complete; the training record's prose sections are authored content, not code placeholders.

**Type consistency:** `_rule_prepass` returns `(text, stats, residue)` (verified in extract.py); `reconstruct`/`invert` share the `(str, dict)` return; `build_jobs` metas carry `corpus/doc_id/R/doc_p` (verified in survival_by_type.py); judge verdict dicts use `label`/`quote` (verified); `_type_sane(entity_type, fill, window)` exists in extract.py (verified). `edit_guard`/`restorable`/`build_target`/`load_reconstructor`/`run_model`/`reconstruct`/`classify_recovery` names are consistent across Tasks 1–6.

**Known limitation to carry into Results:** `edit_guard` is char-diff / substring based (a deliberate ceiling); it bounds inserted content to allowed originals but still permits bounded deletions — if `harm_rate` or `wrong_insert` is nonzero, upgrade to token-constrained decoding (`prefix_allowed_tokens_fn`) before adding capacity elsewhere.
