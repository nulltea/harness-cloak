# Design 3 Reconstructor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local learned reconstructor that recovers the survived-span residue the rule cascade misses — paraphrase-reworded and lossy fill mentions (D-1 semantic, D-3) — by editing `out_p` using the original surfaces in R, pushing survived-span recovery from ~82% toward the client-side ceiling (~100%).

**Architecture:** Residue-targeted constrained edit. The existing deterministic cascade (`invert()`) runs first and resolves A/B/C + alias cases; only its residue generalization entries Q are handed to a flan-t5-base + LoRA seq2seq that reads the cascade's partial output plus a linearized restore-map (fill → original, typed) and rewrites, with a **copy-bias guard** that permits only R's original surfaces as novel content (else falls back to the cascade output — do-no-harm). Training targets are distilled from the Qwen survival judge, which already grounds each reworded mention to a verbatim `out_p` quote.

**Tech Stack:** Python, PyTorch (ROCm host `.venv`), transformers 5.12 + peft 0.19 (LoRA), `google/flan-t5-base`, the `inferdpt.llm` OpenAI-compatible client (gemma out_p pin, Qwen judge pin), rapidfuzz.

## Global Constraints

- **out_p pin (unchanged):** gemma 4 (E4B), temp 0, non-thinking, `RT_BASE_URL=http://localhost:8060/v1`, `max_tokens 1024`; content-addressed cache via `INFERDPT_LLM_CACHE=data/llm_cache`. Reused verbatim from `cloak.train.roundtrip`.
- **Judge pin (target-builder):** `Qwen3.6-35B-A3B`, temp 0, non-thinking, same base_url; llama-swap serves Qwen `-np 1` (serial). Reused verbatim from `scripts/spikes/survival_by_type.py`.
- **Reconstructor model:** `google/flan-t5-base` + LoRA (peft). Local GPU only; never `pip install torch` (see `~/docs/torch-gpu.md`). Checkpoints under `data/models/reconstructor_v1/`.
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

### Task 2: Copy-bias guard (do-no-harm enforcement)

**Files:**
- Modify: `src/cloak/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `copy_bias_guard(source: str, candidate: str, allowed_surfaces: list[str]) -> bool` — True iff every alphabetic word in `candidate` is either present in `source` or in one of `allowed_surfaces` (the R originals for the residue). Used to accept/reject a model rewrite before it replaces the cascade output.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_reconstruct.py
from cloak.reconstruct import copy_bias_guard

def test_guard_accepts_copied_plus_allowed_surface():
    src = "Patient has a disease and takes a drug."
    cand = "Patient has arthritis and takes a drug."   # 'arthritis' is an allowed surface
    assert copy_bias_guard(src, cand, allowed_surfaces=["arthritis"])

def test_guard_rejects_novel_hallucinated_content():
    src = "Patient has a disease."
    cand = "Patient has arthritis in Boston."   # 'Boston' neither copied nor allowed
    assert not copy_bias_guard(src, cand, allowed_surfaces=["arthritis"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_guard_accepts_copied_plus_allowed_surface -v`
Expected: FAIL with `ImportError`/`AttributeError` on `copy_bias_guard`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cloak/reconstruct.py
def _words(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z]+", s)}


def copy_bias_guard(source: str, candidate: str, allowed_surfaces: list[str]) -> bool:
    """do-no-harm: reject a rewrite that introduces content words not copied from `source`
    and not part of an allowed original surface. ponytail: word-level guard with a known
    ceiling — upgrade to token-constrained decoding (prefix_allowed_tokens_fn) only if the
    measured reject/fallback rate is high."""
    permitted = _words(source)
    for surf in allowed_surfaces:
        permitted |= _words(surf)
    return _words(candidate) <= permitted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cloak/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: reconstructor copy-bias guard (do-no-harm)"
```

---

### Task 3: Training-data builder — distill the survival judge

**Files:**
- Create: `scripts/build_reconstructor_data.py`
- Test: `tests/test_reconstruct.py` (target-construction unit only)

**Interfaces:**
- Consumes: `linearize_restore_map`, `splice_at_quote` (Task 1); `cloak.extract._rule_prepass` (returns `(text, stats, residue)`); the judge machinery from `scripts/spikes/survival_by_type.py` (`build_jobs`, `_judge`, `parse_judge`, `JUDGE_TMPL`, `SYSTEM`, `grounded`, `fill_present`); `cloak.train.roundtrip.roundtrip_batch`.
- Produces: JSONL at `data/reconstructor_<corpus>.jsonl`, one row `{"input": str, "target": str, "corpus": str, "doc_id": str, "n_residue": int, "n_edits": int}`. `input` = `<cascade out_final'>\n\n[RESTORE]\n<linearized residue>`; `target` = `input`'s text region with each judge-located reworded mention spliced to its original, or unchanged where the judge abstains (D-4/absent) — this teaches the abstain behavior.

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
    """Splice each located mention's original into text (right-to-left by match position so
    offsets stay valid); entries with a falsy quote are abstains (no edit). Returns
    (target_text, n_edits)."""
    edits = 0
    # apply longest-quote-first to avoid a short quote matching inside a longer one
    for e in sorted([x for x in located if x.get("quote")],
                    key=lambda x: -len(x["quote"])):
        new, ok = splice_at_quote(text, e["quote"], e["surface"])
        if ok:
            text, edits = new, edits + 1
    return text, edits
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
from cloak.reconstruct import linearize_restore_map, build_target
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
    per_corpus: dict[str, list[dict]] = {}

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)  # cascade output
        if not residue:
            continue
        items = "\n".join(f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type","MISC")}]'
                          for i, e in enumerate(residue))
        verdicts = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                              system=SYSTEM), len(residue))
        located = []
        for e, v in zip(residue, verdicts):
            q = v.get("quote") if v.get("label") in ("SURVIVED", "REWORDED") else None
            # only trust a quote grounded in the CASCADE output we will edit
            located.append({"surface": e["surface"],
                            "quote": q if (q and grounded(q, prepass)) else None})
        inp = f"{prepass}\n\n[RESTORE]\n{linearize_restore_map(residue)}"
        target, n_edits = build_target(prepass, located)
        per_corpus.setdefault(m["corpus"], []).append(
            {"input": inp, "target": target, "corpus": m["corpus"],
             "doc_id": m["doc_id"], "n_residue": len(residue), "n_edits": n_edits})

    for corpus, rows in per_corpus.items():
        out = Path(f"data/reconstructor_{corpus}.jsonl")
        out.write_text("\n".join(json.dumps(r) for r in rows))
        edits = sum(r["n_edits"] for r in rows)
        print(f"{corpus}: {len(rows)} docs with residue, {edits} located edits -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then build the data**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v` → PASS (6 tests)
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


def load(path, tok):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    def enc(r):
        x = tok(PROMPT.format(input=r["input"]), truncation=True, max_length=1024)
        y = tok(text_target=r["target"], truncation=True, max_length=1024)
        x["labels"] = y["input_ids"]
        return x
    return Dataset.from_list(rows).map(enc, remove_columns=["input", "target", "corpus",
                                       "doc_id", "n_residue", "n_edits"])


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
- Consumes: `_rule_prepass` (`cloak.extract`), `linearize_restore_map`, `copy_bias_guard`, `_finalize` (`cloak.extract`).
- Produces: `load_reconstructor(path) -> obj`; `run_model(obj, prompt: str) -> str`; `reconstruct(out_p: str, R: list[dict], model=None) -> tuple[str, dict]` — cascade first; if residue and a model is given, run the model on `<prepass>\n\n[RESTORE]\n<map>`, accept via `copy_bias_guard` (else keep cascade output), then `_finalize`. Signature-compatible with `invert()` so it drops into any caller.

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
    # model hallucinates 'Boston' -> guard rejects -> cascade output kept (still 'a disease')
    text, stats = reconstruct(out_p, R, model=_StubModel("Patient has arthritis in Boston."))
    assert "Boston" not in text
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
        allowed = [e["surface"] for e in residue]
        if copy_bias_guard(prepass, cand, allowed):
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
Expected: PASS (8 tests)

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
- Produces: `results/reconstructor_eval.json` — per-type mention-anchored recovery of survived spans for cascade vs cascade+reconstructor, plus false-substitution count. Metric helper `recovered_at_quote(out_final, quote, surface) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_reconstruct.py
def test_recovered_at_quote():
    from cloak.reconstruct import recovered_at_quote
    # original now stands where the reworded mention was
    assert recovered_at_quote("filed in January 13th 1982 today", "Early 1980s", "January 13th 1982")
    # neither the original nor an edit present -> not recovered
    assert not recovered_at_quote("filed in Early 1980s today", "Early 1980s", "January 13th 1982")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py::test_recovered_at_quote -v`
Expected: FAIL with `ImportError` on `recovered_at_quote`

- [ ] **Step 3: Implement metric + eval script**

```python
# add to src/cloak/reconstruct.py
def recovered_at_quote(out_final: str, quote: str, surface: str) -> bool:
    """Mention-anchored recovery: the original `surface` is present and the reworded
    `quote` no longer stands verbatim (it was edited)."""
    return _norm(surface) in _norm(out_final) and not (
        quote and _norm(quote) in _norm(out_final) and _norm(surface) not in _norm(quote))
```

```python
# scripts/spikes/reconstructor_eval.py
"""Mention-anchored recovery of survived spans: cascade (invert) vs cascade+reconstructor.
Held-out corpus (train clinical -> eval lexsum). Reports per-type recovery + false subs.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/spikes/reconstructor_eval.py \
       --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
       --corpora lexsum --n-docs 80 --ckpt data/models/reconstructor_v1
"""
import argparse, json
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, exact_present,
                              fill_present, SYSTEM, JUDGE_TMPL)
from cloak.extract import invert
from cloak.reconstruct import reconstruct, load_reconstructor, recovered_at_quote
from cloak.train.roundtrip import roundtrip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", required=True); ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge(); model = load_reconstructor(args.ckpt)
    rows = {}   # type -> {survived, rec_cascade, rec_recon, false_recon}

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
        for e, vv in zip(gens, v):
            q = vv.get("quote"); lbl = vv.get("label", "ABSENT")
            surv = exact_present(e["replacement"], out_p) or (
                lbl in ("SURVIVED", "REWORDED") and grounded(q, out_p))
            # count only substituted-content survivals (exclude leaked-only)
            if not surv or (not fill_present(e["replacement"], out_p)
                            and exact_present(e["surface"], out_p)):
                continue
            t = e.get("type", "MISC"); r = rows.setdefault(t, dict(survived=0, rec_cascade=0,
                                                                   rec_recon=0, false_recon=0))
            r["survived"] += 1
            r["rec_cascade"] += int(recovered_at_quote(casc, q, e["surface"]))
            r["rec_recon"] += int(recovered_at_quote(recon, q, e["surface"]))
            # false sub: reconstructor put the surface where the judge said ABSENT
            if lbl == "ABSENT" and e["surface"] in recon and e["surface"] not in out_p:
                r["false_recon"] += 1

    report = {"ckpt": args.ckpt, "corpora": args.corpora, "rows": rows,
              "totals": {k: sum(r[k] for r in rows.values())
                         for k in ("survived", "rec_cascade", "rec_recon", "false_recon")}}
    Path("results/reconstructor_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test, then the eval (held-out lexsum)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reconstruct.py -v` → PASS (9 tests)
Then (check `pgrep -af train_pii`):
```bash
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes .venv/bin/python -u \
  scripts/spikes/reconstructor_eval.py --env data/ranker_env_pilot.json \
  --arms data/task_arms_pilot.json --corpora lexsum --n-docs 80 \
  --ckpt data/models/reconstructor_v1 > results/reconstructor_eval.log 2>&1
```
Expected: `results/reconstructor_eval.json` with per-type `rec_recon >= rec_cascade` and `false_recon == 0` (success: reconstructor lifts recovery on D-1/D-3 with zero false substitutions). Any `false_recon > 0` is a hard fail — tighten the guard or the abstain training data.

- [ ] **Step 5: Fill the training record Results & commit**

Set `status: done` in `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md`; fill Results with the measured per-type recovery (report the win AND any regression/false-sub, per empirical-honesty). Cross-link the survived-recovery design doc and the experiment record.

```bash
git add scripts/spikes/reconstructor_eval.py src/cloak/reconstruct.py tests/test_reconstruct.py \
        research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md
git commit -m "feat: reconstructor eval (mention-anchored recovery vs cascade) + v1 results"
```

---

## Self-Review

**Spec coverage:** residue-targeted edit (Task 5 `reconstruct`), judge-distilled targets (Task 3), copy-bias do-no-harm (Task 2 guard + Task 5 fallback), abstain on D-4/absent (Task 3 unchanged-target + Task 6 false_recon gate), held-out eval per corpus (Task 6), training record spec-then-results (Tasks 4/6). D-2 acronym/alias is handled by the deterministic proposers in the companion survived-recovery design, not this model — noted so it is not a gap.

**Placeholder scan:** none — every code step is complete; the training record's prose sections are authored content, not code placeholders.

**Type consistency:** `_rule_prepass` returns `(text, stats, residue)` (verified in extract.py); `reconstruct`/`invert` share the `(str, dict)` return; `build_jobs` metas carry `corpus/doc_id/R/doc_p` (verified in survival_by_type.py); judge verdict dicts use `label`/`quote` (verified). `load_reconstructor`/`run_model`/`reconstruct`/`recovered_at_quote`/`build_target` names are consistent across Tasks 3–6.

**Known limitation to carry into Results:** the copy-bias guard is word-level (`copy_bias_guard`), a deliberate ceiling — if the measured `gen_recon_rejected` rate is high, upgrade to token-constrained decoding (`prefix_allowed_tokens_fn`) before adding capacity elsewhere.
