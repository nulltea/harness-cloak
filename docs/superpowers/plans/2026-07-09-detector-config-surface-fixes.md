# Detector Config-Surface Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six confirmed detector pipeline defects (chunk-boundary span loss, widest-wins dedupe, clinical-"RN" pronoun drop, coref identity collapse, Presidio vitals misfires, encoder-window overflow) found by `scripts/spikes/detector_config_surface_probes.py`.

**Architecture:** All changes live in `src/cloak/detect.py` (plus a two-line wiring change in `scripts/build_mined_lattice_profiles.py`). Pure-function fixes first (`_chunks`, `_dedupe`, `coref_chains`), then a small per-corpus `DetectorProfile` mechanism (stop-word set + custom-recognizer selection), then the encoder-window guard wired into both consumers. No new dependencies.

**Tech Stack:** Python 3.12, pytest (tests in `src/cloak/tests/`), GLiNER, presidio-analyzer. Tests here are model-free (pure functions + profile tables); nothing in this plan loads GLiNER or spaCy.

## Global Constraints

- **Default behavior stays bit-identical**: `Detector()` with no arguments must behave exactly as today (same stop-word set including `rn`/`ngl`, same custom recognizers, same threshold/chunking) except for the five bug fixes themselves. Empirical-honesty rule: no silent shift of the eval operating point. Default profile is `"reddit"` (= status quo).
- **No plan/doc-internal identifiers in code** (CLAUDE.md naming rule): name things after what they do (`DetectorProfile`, `max_words`), never "task 3" / "defect 2".
- Tests go in `src/cloak/tests/test_detect_<area>.py`; run with `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/<file> -v`. Do NOT run the whole suite (some tests load models); run only the named test files.
- Commit messages: conventional style (`fix(detect): ...`), one commit per task, `git add` only the files the task names. The working tree has unrelated modified data files (a live run writes them) — never `git add -A`.
- Do not modify `scripts/spikes/detector_config_surface_probes.py` (it documents the pre-fix behavior) and do not touch `results/` or `data/`.

---

### Task 1: `_chunks` — never cut mid-word, optional word-count cap

**Files:**
- Modify: `src/cloak/detect.py:122-131` (`_chunks`)
- Test: `src/cloak/tests/test_detect_chunks.py` (create)

**Interfaces:**
- Produces: `_chunks(text: str, max_chars: int = 1200, max_words: int | None = None)` yielding `(offset, chunk)`; `text[offset:offset+len(chunk)] == chunk` always holds. Later tasks pass `max_words` from the model config.

Current bug: when no `\n`/`. ` falls in the window's second half, the chunk is hard-cut at `max_chars`, splitting a word (and any entity in it) across chunks. Also no guard against a chunk exceeding the encoder's word window.

- [ ] **Step 1: Write the failing tests**

```python
"""_chunks: boundary placement and word-count cap."""
from cloak.detect import _chunks


def _reassemble(text, chunks):
    assert all(text[off:off + len(c)] == c for off, c in chunks)


def test_no_midword_cut():
    # 1195 filler chars then a name: the old code hard-cut inside "Sarah"
    text = "x" * 1195 + " Sarah Johnson was seen"
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert len(chunks) == 2
    # every chunk boundary lands on whitespace: no chunk ends or starts mid-word
    assert chunks[0][1].endswith(" ") or chunks[1][1].startswith(" ") or not chunks[0][1][-1].isalnum() or not chunks[1][1][0].isalnum()
    assert "Sarah Johnson" in chunks[1][1]


def test_sentence_cut_still_preferred():
    text = "y" * 900 + " Seen at the clinic. " + "z" * 400
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert chunks[0][1].rstrip().endswith("clinic.")


def test_unbroken_text_still_terminates():
    # a single 5000-char token cannot be word-preserved; hard cut is acceptable
    text = "x" * 5000
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert sum(len(c) for _, c in chunks) == 5000


def test_max_words_caps_chunks():
    # spaced-out OCR-style text: 800 single-char words in one 1200-char window
    text = " ".join("a" for _ in range(800))
    chunks = list(_chunks(text, max_words=100))
    _reassemble(text, chunks)
    assert all(len(c.split()) <= 100 for _, c in chunks)
    assert "".join(c for _, c in chunks).replace(" ", "") == "a" * 800


def test_max_words_none_is_noop():
    text = "one two three. " * 200
    assert list(_chunks(text)) == list(_chunks(text, max_words=None))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_chunks.py -v`
Expected: `test_no_midword_cut` and `test_max_words_caps_chunks` FAIL (mid-word cut / TypeError for `max_words`); the others may pass.

- [ ] **Step 3: Implement**

Replace `_chunks` in `src/cloak/detect.py` with:

```python
def _chunks(text: str, max_chars: int = 1200, max_words: int | None = None):
    """Split on line/sentence boundaries into ~max_chars windows; yield (offset, chunk).

    Never cuts mid-word: if no newline/sentence break falls in the window's second half,
    back off to the last whitespace instead of a hard character cut (a hard cut splits the
    entity under it across chunks). max_words re-splits any chunk whose whitespace token
    count exceeds the encoder window (spaced-out OCR/ASR text inflates tokens ~2x per char;
    gliner-pii-large has max_len=768 vs 2048 for the base models).
    """
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", pos, end), text.rfind(". ", pos, end))
            if cut > pos + max_chars // 2:
                end = cut + 1
            else:
                ws = text.rfind(" ", pos + max_chars // 2, end)
                if ws > pos:
                    end = ws + 1  # word boundary, not a mid-word character cut
        chunk = text[pos:end]
        if max_words and len(chunk.split()) > max_words:
            words = list(re.finditer(r"\S+", chunk))
            for i in range(0, len(words), max_words):
                last = words[min(i + max_words, len(words)) - 1]
                yield pos + words[i].start(), chunk[words[i].start():last.end()]
        else:
            yield pos, chunk
        pos = end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_chunks.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/detect.py src/cloak/tests/test_detect_chunks.py
git commit -m "fix(detect): _chunks never cuts mid-word; optional max_words encoder-window cap"
```

---

### Task 2: `_dedupe` — type-aware overlap resolution + Presidio pattern tagging

**Files:**
- Modify: `src/cloak/detect.py:232-238` (`_dedupe`) and `detect.py:220-226` (Presidio loop in `Detector.detect`)
- Test: `src/cloak/tests/test_detect_dedupe.py` (create)

**Interfaces:**
- Consumes: `Span` dataclass as-is.
- Produces: `_dedupe(spans) -> list[Span]` (start-sorted). Span `source` gains a third value: `"presidio-pattern"` for non-spaCy Presidio recognizers (regex/pattern based: email, phone, SSN, IBAN, the custom REF_CODE/MONEY, ...); `"presidio"` now means spaCy-NER-backed results only.

Current bug: overlap resolution keeps the *widest* span regardless of score, source, or type — a 0.31 wide GLiNER MISC span swallows a 0.99 Presidio SSN CODE span. Policy decided: same-type overlaps keep the widest (extent disagreement); cross-type conflicts go to the higher score, with pattern-based Presidio hits given an effective floor of 0.9 (their fixed pattern scores, 0.4–0.6, are not comparable to GLiNER probabilities).

- [ ] **Step 1: Write the failing tests**

```python
"""_dedupe: same-type widest, cross-type score-first with a pattern-hit floor."""
from cloak.detect import Span, _dedupe


def _s(start, end, typ, score, source="gliner"):
    return Span(start, end, "x" * (end - start), typ, score, source)


def test_same_type_keeps_widest():
    wide, narrow = _s(0, 20, "PERSON", 0.4), _s(5, 10, "PERSON", 0.95)
    assert _dedupe([narrow, wide]) == [wide]


def test_cross_type_higher_score_wins():
    misc = _s(0, 30, "MISC", 0.31)
    code = _s(10, 21, "CODE", 0.99)
    assert _dedupe([misc, code]) == [code]


def test_presidio_pattern_gets_score_floor():
    # pattern recognizers report fixed low scores (0.4-0.6); they must not lose
    # a cross-type conflict to a mid-confidence gliner span
    misc = _s(0, 30, "MISC", 0.7)
    ssn = _s(10, 21, "CODE", 0.6, source="presidio-pattern")
    assert _dedupe([misc, ssn]) == [ssn]


def test_presidio_spacy_gets_no_floor():
    misc = _s(0, 30, "MISC", 0.7)
    spacy_loc = _s(10, 21, "LOC", 0.6, source="presidio")
    assert _dedupe([misc, spacy_loc]) == [misc]


def test_non_overlapping_all_kept_start_sorted():
    a, b = _s(10, 20, "PERSON", 0.9), _s(0, 5, "CODE", 0.5)
    assert _dedupe([a, b]) == [b, a]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_dedupe.py -v`
Expected: `test_cross_type_higher_score_wins` and `test_presidio_pattern_gets_score_floor` FAIL (widest currently wins).

- [ ] **Step 3: Implement**

Replace `_dedupe` in `src/cloak/detect.py` with:

```python
def _dedupe(spans: list[Span]) -> list[Span]:
    """Overlap resolution. Same-type overlaps: keep the widest (extent disagreement over one
    entity). Cross-type conflicts: higher score wins; pattern-based Presidio hits (fixed regex
    scores 0.4-0.6, not comparable to GLiNER probabilities) get an effective floor of 0.9.
    """
    def eff(s: Span) -> float:
        return max(s.score, 0.9) if s.source == "presidio-pattern" else s.score

    within_type: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), -s.score, s.start)):
        if not any(s.type == o.type and s.start < o.end and o.start < s.end for o in within_type):
            within_type.append(s)
    out: list[Span] = []
    for s in sorted(within_type, key=lambda s: (-eff(s), -(s.end - s.start), s.start)):
        if not any(s.start < o.end and o.start < s.end for o in out):
            out.append(s)
    return sorted(out, key=lambda s: s.start)
```

In `Detector.detect`, tag pattern-based Presidio results. Replace the Presidio append:

```python
        for r in self.presidio.analyze(text=text, language="en"):
            if r.entity_type in PRESIDIO_MAP:
                t = PRESIDIO_MAP[r.entity_type]
                if self.fine_dem and t == "DEM":
                    continue   # fine-dem: GLiNER's learned fine leaves own demographics; drop Presidio's
                               # coarse NRP->DEM (keeps relabel_dem training/eval-only, inference pure-model).
                rec = (r.recognition_metadata or {}).get("recognizer_name", "")
                src = "presidio" if rec == "SpacyRecognizer" else "presidio-pattern"
                spans.append(Span(r.start, r.end, text[r.start:r.end], t, r.score, src))
```

Also update the `Span.source` docstring comment: `source: str    # "gliner" | "presidio" (spaCy NER) | "presidio-pattern"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_dedupe.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Check no other code matches on the exact string `"presidio"`**

Run: `grep -rn '"presidio"' src scripts --include="*.py"`
Any consumer doing `s.source == "presidio"` must be changed to `s.source.startswith("presidio")`. (As of planning, the only hits are detect.py itself; verify.)

- [ ] **Step 6: Commit**

```bash
git add src/cloak/detect.py src/cloak/tests/test_detect_dedupe.py
git commit -m "fix(detect): type-aware overlap resolution; tag pattern-based presidio spans"
```

---

### Task 3: `coref_chains` — stricter aliasing, no surname-only identity collapse

**Files:**
- Modify: `src/cloak/detect.py:241-262` (`coref_chains`)
- Test: `src/cloak/tests/test_detect_coref.py` (create)

**Interfaces:**
- Consumes/Produces: `coref_chains(text, spans) -> list[Span]` (sets `s.chain`, signature unchanged). Consumer: `cloak.substitute.substitute()` uses chain ids for placeholder consistency.

Current bug: any same-type token overlap joins a chain, so "Anna Smith" and "Peter Smith" (shared surname) collapse into one chain → one placeholder → two people merged in the anonymized output. Decided policy: chain only on (a) whole-token containment either way ("Anna Smith" ⊇ "Anna") or (b) matching first token ("Anna Smith" ~ "Anna S."); a bare single-token mention that is a *trailing* token of an existing chain member (bare "Smith") joins the **most recent** matching chain.

- [ ] **Step 1: Write the failing tests**

```python
"""coref_chains: containment/first-token aliasing; no surname-only merges."""
from cloak.detect import Span, coref_chains


def _spans(text, *surfaces):
    out, pos = [], 0
    for surf in surfaces:
        i = text.index(surf, pos)
        out.append(Span(i, i + len(surf), surf, "PERSON", 0.9, "gliner"))
        pos = i + len(surf)
    return out


def test_shared_surname_does_not_merge():
    text = "Anna Smith met Peter Smith"
    spans = coref_chains(text, _spans(text, "Anna Smith", "Peter Smith"))
    assert spans[0].chain != spans[1].chain


def test_containment_merges():
    text = "Anna Smith arrived. Anna spoke."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Anna"))
    assert spans[0].chain == spans[1].chain


def test_first_token_match_merges():
    text = "Anna Smith arrived. Anna S. spoke."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Anna S."))
    assert spans[0].chain == spans[1].chain


def test_bare_trailing_token_joins_most_recent():
    text = "Anna Smith met Peter Smith. Later Smith left."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Peter Smith", "Smith"))
    assert spans[2].chain == spans[1].chain  # most recent Smith chain (Peter's)
    assert spans[0].chain != spans[1].chain


def test_bare_member_is_not_a_containment_bridge():
    # regression: a stored bare "Smith" member must not transitively merge two full names
    text = "Anna Smith testified. Smith spoke. Peter Smith arrived."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Smith", "Peter Smith"))
    assert spans[0].chain == spans[1].chain      # bare mention joins most recent (Anna's)
    assert spans[2].chain != spans[0].chain      # Peter stays a distinct identity


def test_leading_bare_token_does_not_bridge():
    text = "Smith left early. Anna Smith stayed. Peter Smith arrived."
    spans = coref_chains(text, _spans(text, "Smith", "Anna Smith", "Peter Smith"))
    assert spans[1].chain != spans[2].chain      # no merge through the bare seed


def test_different_types_never_merge():
    text = "Smith worked at Smith Hospital"
    s = _spans(text, "Smith")
    i = text.index("Smith Hospital")
    s.append(Span(i, i + len("Smith Hospital"), "Smith Hospital", "ORG", 0.9, "gliner"))
    spans = coref_chains(text, s)
    assert spans[0].chain != spans[1].chain
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_coref.py -v`
Expected: `test_shared_surname_does_not_merge` FAILS (current any-token-overlap rule merges); others may pass.

- [ ] **Step 3: Implement**

Replace `coref_chains` in `src/cloak/detect.py` with:

```python
def coref_chains(text: str, spans: list[Span]) -> list[Span]:
    """Attach chain ids by surface aliasing. Same-type spans chain only when one surface
    contains the other as whole tokens ("Anna Smith" ~ "Anna") or their first tokens match
    ("Anna Smith" ~ "Anna S."). A bare single-token mention that is a non-first token of an
    existing member (bare "Smith") joins the MOST RECENT matching chain. Any-token overlap
    is deliberately not enough: it merged distinct people sharing a surname into one
    placeholder.

    ponytail: string-alias coref — fastcoref 2.1.6 is incompatible with transformers 5.12
    (FCorefModel hits removed modeling internals). Upgrade to a real coref model for the TAB
    pass, where nominal anaphora ("the applicant") matters.
    """
    def toks(surface: str) -> list[str]:
        return [t for t in surface.lower().split() if len(t) > 2]

    def aliases(a: list[str], b: list[str]) -> bool:
        if not a or not b:
            return False
        # containment requires BOTH sides multi-token: a bare stored member ("smith") must not
        # become a containment magnet that transitively re-merges distinct people. Single-token
        # aliasing is handled exclusively by first-token match and the bare-mention clause below.
        if min(len(a), len(b)) >= 2:
            sa, sb = set(a), set(b)
            if sa <= sb or sb <= sa:
                return True
        return a[0] == b[0]

    chains: list[tuple[str, list[list[str]]]] = []  # (type, member token lists)
    for s in sorted(spans, key=lambda s: s.start):
        st = toks(s.text)
        s.chain = -1
        for ci in reversed(range(len(chains))):      # most recent chain wins
            ctype, members = chains[ci]
            if ctype != s.type:
                continue
            if any(aliases(st, m) for m in members) or (
                    len(st) == 1 and any(st[0] in m for m in members)):
                s.chain = ci
                members.append(st)
                break
        if s.chain < 0:
            chains.append((s.type, [st]))
            s.chain = len(chains) - 1
    return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_coref.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Run the existing consumer tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_substitute_prepass.py -v`
Expected: PASS (they monkeypatch `coref_chains`, but confirm no import breakage).

- [ ] **Step 6: Commit**

```bash
git add src/cloak/detect.py src/cloak/tests/test_detect_coref.py
git commit -m "fix(detect): coref aliasing requires containment or first-token match, not any shared token"
```

---

### Task 4: Per-corpus detector profiles (stop-word sets + custom recognizers)

**Files:**
- Modify: `src/cloak/detect.py` — `_PRONOUNS` block (lines ~107-108), `Detector.__init__` (recognizer registration, lines ~194-210), `Detector.detect` (the pronoun filter, line ~228)
- Test: `src/cloak/tests/test_detect_profiles.py` (create)

**Interfaces:**
- Produces: `DetectorProfile` frozen dataclass (`name: str`, `slang_stop_words: bool`, `custom_recognizers: bool`); module dict `PROFILES = {"reddit": ..., "legal": ..., "clinical": ...}`; `Detector(..., profile: str = "reddit")`. `Detector.stop_words: frozenset[str]` used by `detect()`.
- Constraint: `Detector()` (no args) must keep today's exact behavior → default `profile="reddit"` (slang stop words AND custom recognizers, exactly the current config).

Current bug: Reddit-era choices are hardcoded for all corpora — the `_PRONOUNS` stop set contains chat slang `rn`/`ngl`, so a detected clinical "RN" (registered nurse) span is silently deleted; the custom `REF_CODE`/`MONEY` regexes match blood pressure `120/80`, year ranges `2021/22`, and `5k`/`10M`. Decided design: per-corpus profiles select the stop set and recognizer set. `reddit` = slang + recognizers (status quo, default); `legal` = no slang, keep recognizers (TAB case refs like `36110/97` and money amounts are real CODE/QUANTITY there); `clinical` = no slang, no custom recognizers (vitals/ranges dominate; case refs and money amounts are rare).

- [ ] **Step 1: Write the failing tests**

```python
"""DetectorProfile: stop-word and recognizer selection per corpus."""
import pytest

from cloak.detect import PROFILES, _stop_words


def test_reddit_profile_is_status_quo():
    p = PROFILES["reddit"]
    assert p.slang_stop_words and p.custom_recognizers
    sw = _stop_words(p)
    assert "rn" in sw and "ngl" in sw and "she" in sw


def test_clinical_profile_keeps_rn_spans():
    p = PROFILES["clinical"]
    assert not p.slang_stop_words and not p.custom_recognizers
    sw = _stop_words(p)
    assert "rn" not in sw and "ngl" not in sw and "she" in sw


def test_legal_profile_no_slang_keeps_recognizers():
    p = PROFILES["legal"]
    assert not p.slang_stop_words and p.custom_recognizers
    assert "rn" not in _stop_words(p)


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        PROFILES["nosuch"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_profiles.py -v`
Expected: FAIL with ImportError (`PROFILES`, `_stop_words` don't exist).

- [ ] **Step 3: Implement**

In `src/cloak/detect.py`, replace the `_PRONOUNS` block with:

```python
_PRONOUNS = frozenset({"i", "me", "my", "mine", "you", "your", "he", "him", "his", "she",
                       "her", "it", "its", "we", "us", "our", "they", "them", "their"})
_SLANG_STOP = frozenset({"rn", "ngl"})  # chat slang; but "RN" = registered nurse in clinical text


@dataclass(frozen=True)
class DetectorProfile:
    """Per-corpus detector configuration: which stop words suppress detected spans and
    whether the custom pattern recognizers (REF_CODE, MONEY) are registered. Those regexes
    are right for reddit/legal text but misfire on clinical vitals (120/80) and ranges."""
    name: str
    slang_stop_words: bool
    custom_recognizers: bool


PROFILES = {
    "reddit": DetectorProfile("reddit", slang_stop_words=True, custom_recognizers=True),
    "legal": DetectorProfile("legal", slang_stop_words=False, custom_recognizers=True),
    "clinical": DetectorProfile("clinical", slang_stop_words=False, custom_recognizers=False),
}


def _stop_words(profile: DetectorProfile) -> frozenset[str]:
    return _PRONOUNS | _SLANG_STOP if profile.slang_stop_words else _PRONOUNS
```

In `Detector.__init__`, add the keyword `profile: str = "reddit"` (default = status quo; changing it moves the eval operating point). After `self.presidio = AnalyzerEngine()`:

```python
        self.profile = PROFILES[profile]
        self.stop_words = _stop_words(self.profile)
        if self.profile.custom_recognizers:
            from presidio_analyzer import Pattern, PatternRecognizer
            ... (the existing two add_recognizer calls, unchanged, indented under the if)
```

In `Detector.detect`, change the filter line to use the instance set:

```python
        spans = [s for s in spans  # pure symbol/emoji spans or bare stop words: never identifiers
                 if re.search(r"[A-Za-z0-9]", s.text) and s.text.lower() not in self.stop_words]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_profiles.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Verify no stale `_PRONOUNS` references**

Run: `grep -rn "_PRONOUNS\|_SLANG_STOP\|stop_words" src scripts --include="*.py" | grep -v tests`
Expected: only definitions in detect.py and the `self.stop_words` uses; `scripts/spikes/detector_config_surface_probes.py` still imports `_PRONOUNS` — it must keep importing cleanly (the name still exists), but do not edit it.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/detect.py src/cloak/tests/test_detect_profiles.py
git commit -m "feat(detect): per-corpus detector profiles select stop words and custom recognizers"
```

---

### Task 5: Encoder-window guard wired into Detector and the miner

**Files:**
- Modify: `src/cloak/detect.py` (`Detector.__init__` + `Detector.detect`)
- Modify: `scripts/build_mined_lattice_profiles.py:212-253` (`detect_clinical_spans`)
- Test: `src/cloak/tests/test_detect_window_guard.py` (create)

**Interfaces:**
- Consumes: `_chunks(..., max_words=...)` from the chunking task.
- Produces: `_encoder_max_words(gliner) -> int | None` in `cloak.detect` — reads `gliner.config.max_len` and returns a 0.9-margin word cap (`int(max_len * 0.9)`), or `None` when the attribute is missing. `Detector.detect` and the miner's `detect_clinical_spans` pass it to `_chunks`.

Why: GLiNER's `max_len` counts words, and models differ silently — base/fine-tune 2048, `gliner-pii-large` 768. A 1200-char chunk of spaced-out OCR/ASR text ("b u n") can exceed 768 words, and the tail of the chunk is then silently unscanned.

- [ ] **Step 1: Write the failing test**

```python
"""_encoder_max_words: derive the per-model word cap from gliner config."""
from types import SimpleNamespace

from cloak.detect import _encoder_max_words


def test_reads_max_len_with_margin():
    g = SimpleNamespace(config=SimpleNamespace(max_len=768))
    assert _encoder_max_words(g) == 691  # int(768 * 0.9)


def test_missing_max_len_returns_none():
    g = SimpleNamespace(config=SimpleNamespace())
    assert _encoder_max_words(g) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_window_guard.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

In `src/cloak/detect.py`, add next to `_chunks`:

```python
def _encoder_max_words(gliner) -> int | None:
    """Word cap for _chunks from the model's window. gliner max_len counts words and varies
    by model (base/fine-tune 2048, gliner-pii-large 768); overflow is silently truncated by
    the encoder, so chunks must stay under it. 0.9 margin for the label prompt overhead."""
    max_len = getattr(gliner.config, "max_len", None)
    return int(max_len * 0.9) if max_len else None
```

In `Detector.__init__`, after the GLiNER model is loaded: `self.max_words = _encoder_max_words(self.gliner)`.
In `Detector.detect`, change the chunking line:

```python
        offsets, texts = zip(*_chunks(text, max_words=self.max_words)) if text.strip() else ((), ())
```

In `scripts/build_mined_lattice_profiles.py` `detect_clinical_spans`, import `_encoder_max_words` alongside `_chunks`, and after `gliner = GLiNER.from_pretrained(model)`:

```python
    max_words = _encoder_max_words(gliner)
```

and change the chunk collection line to:

```python
        chunk_rows.extend((doc["id"], chunk_text) for _, chunk_text in _chunks(doc["text"], max_words=max_words))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_detect_window_guard.py src/cloak/tests/test_detect_chunks.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the miner's existing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_build_mined_lattice_profiles.py -v`
Expected: PASS (pure-function tests; `detect_clinical_spans` is not exercised, but the import must stay clean).

- [ ] **Step 6: Commit**

```bash
git add src/cloak/detect.py scripts/build_mined_lattice_profiles.py src/cloak/tests/test_detect_window_guard.py
git commit -m "fix(detect): cap chunk word count to the model's encoder window (gliner-pii-large is 768)"
```

---

## Residual risks (documented, deliberately not fixed here)

- ~~A multi-word entity can still split when the chunk boundary falls between its words~~ — FIXED
  post-review: fallback (non-sentence) cuts now overlap the next window by `overlap_chars=200` on a
  word boundary; `_dedupe` merges the duplicate detections. Sentence cuts stay contiguous
  (normally punctuated prose chunks identically). Measured: pipeline red tests 4/7 → 0/7.
- `PRESIDIO_MAP` still silently drops unmapped Presidio types (URL deliberate) and `fine_dem` still drops Presidio NRP — both are documented decisions, out of scope.
- The deprecated `batch_predict_entities` call sites (detect.py, miner, red tests) stay — migrating to `GLiNER.inference` is a separate upgrade with its own eval re-run.
- Miner label schema vs production label schema divergence is the noise-investigation's fix, not this plan's.
