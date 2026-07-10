"""Ladder + decision probe generation (training-task-env spec, Probe generation section).

Ladder probes: per lattice-bearing span, one question per rung (rung 0 = the exact surface,
rung l = the l-th lattice level). Gold at rung l is that rung's phrase; scoring accepts any
finer answer via the acceptance set (entail_score) — deterministic, no NLI on the reward path.
Decision probes: per doc (clinical/lexsum), multiple-choice questions a consumer of the output
must answer, gold recovered from the ceiling output (ceiling agreement).

Teacher-cached like cloak.train.probes: entries tagged {teacher, pv}; other-teacher /
other-pv entries are ignored and regenerated, never mixed.
"""
import json
import hashlib
import random
import re
from pathlib import Path

from cloak.lattice import is_type_name_phrase
from cloak.train.probes import TEACHER_MODEL
from cloak.train.reward import canon, fact_score

# Semantically-empty fills ("something", "a disorder") are legitimate lattice actions but
# useless probe golds — a question whose best answer carries no content grades nothing, and
# such fills SHOULD earn no semantic-tier credit over a placeholder. No rung question is kept
# for them (measured: mammogram->"something" in results/ladder_probe_gen_test.json).
_EMPTY_GOLDS = {"something", "a thing", "an issue", "a disorder", "a condition",
                "a physical condition", "a problem"}


def _empty_gold(phrase: str) -> bool:
    f = re.sub(r"\s+", " ", canon(phrase)).strip().rstrip(".")
    return f in _EMPTY_GOLDS or is_type_name_phrase(phrase)

LADDER_CACHE = Path("data/ladder_probes.json")
DECISION_CACHE = Path("data/decision_probes.json")
LADDER_PV = 1
DECISION_PV = 1

OUTPUT_KIND = {"aci": "clinical note", "mts": "clinical note", "clinical": "clinical note",
               "lexsum": "case summary", "wikibio": "biography summary",
               "enron": "email reply", "aeslc": "email subject line",
               "qmsum": "meeting summary"}

DECISION_KINDS = {
    "clinical": ("referral routing, medication continue/adjust/stop, follow-up interval, "
                 "or appropriateness of a plan element"),
    "lexsum": ("likely prevailing party, remedy type, procedural posture, or which court's "
               "rules govern"),
}
DECISION_KINDS["aci"] = DECISION_KINDS["mts"] = DECISION_KINDS["clinical"]

LADDER_PROMPT = """You write probe questions used to grade how well a {output_kind} preserves \
facts at different levels of detail.

A {output_kind} will be written from the document below. Some facts may appear in it only in \
a generalized form. For the target fact you are given its generalization ladder: the exact \
value first, then successively broader truthful descriptions.

For EACH rung, write ONE question that:
1. has exactly that rung's phrase as its best answer when the {output_kind} states the fact \
at that rung — ask about the PROPERTY the rung expresses, not the specific value;
2. does not contain the exact value, any finer rung's phrase, or close synonyms of them;
3. identifies which fact it asks about through surrounding circumstances (what it is treated \
with, its role in the document, who raised it) — never through the fact itself;
4. is a wh- question with a short-phrase answer; no yes/no questions;
5. is answerable from the {output_kind} alone, by a reader who never saw the document.

Document:
{doc}

Target fact: "{surface}"   (type: {type}; appears in: "{sentence}")
Ladder rungs, exact -> broad:
{rungs}

Reply ONLY with a JSON list: [{{"rung": 0, "q": "...", "a": "<that rung's phrase>"}}, ...]"""

DECISION_PROMPT = """You design decision checks that grade whether a {output_kind} supports \
the decisions its readers must make.

Below are a document and the {output_kind} written from it. Write up to {k} decision \
questions that a professional reading ONLY the {output_kind} would need to answer \
({decision_kinds}). For each question:
1. the correct answer must be determinable from the {output_kind} alone;
2. give 3-5 plausible answer options, exactly one correct;
3. the decision must turn on the substantive content, never on names, dates, or other \
administrative details;
4. quote the exact document phrases the decision depends on.

Document:
{doc}

The {output_kind}:
{out_hi}

Reply ONLY with a JSON list:
[{{"q": "...", "options": ["...", "..."], "gold": "...", "depends_on": ["...", "..."]}}]"""

_STOP = {"a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "with", "by"}
_YESNO = re.compile(r"^(is|are|was|were|does|did|do|has|have|had|can|could|should|would|will"
                    r"|may|might)\b", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", canon(text)) if w not in _STOP}


def rung_phrases(surface: str, levels: list[str]) -> list[str]:
    """rungs[0] = exact surface, rungs[l] = l-th lattice level (specific -> broad)."""
    return [surface, *levels]


def span_levels(span: dict) -> list[str]:
    """Ladder levels of an env span, specific -> broad.

    Drops the keep action (fill == surface) and sorts the remaining level actions by
    ascending anonymity-set size. Env action lists are not guaranteed to be sorted.
    """
    surface = canon(span.get("surface", ""))
    acts = [a for a in span.get("actions", [])
            if a.get("mode") == "level" and canon(a.get("fill") or "") != surface]
    acts.sort(key=lambda a: a.get("aset") or 0)
    return [a["fill"] for a in acts]


def entail_score(answer: str, rungs: list[str], rung: int) -> float:
    """Acceptance-set scoring: an answer at or finer than the rung counts."""
    return max(fact_score(answer, a) for a in rungs[:rung + 1])


def lint_rung(q: str, rungs: list[str], rung: int) -> bool:
    """Leakage + form lint. Rejects: non-wh / yes-no / overlong questions; questions giving
    away the full gold phrase; questions containing any token distinctive of a finer rung
    (a token not already in the rung's own gold — generic nouns like 'condition' shared with
    the gold are allowed, 'hypothyroidism' is not)."""
    q = (q or "").strip()
    if not q.endswith("?") or len(q) > 200 or _YESNO.match(q):
        return False
    qt = _tokens(q)
    gold_t = _tokens(rungs[rung])
    if gold_t and gold_t <= qt:
        return False
    finer_t = set().union(*(_tokens(r) for r in rungs[:rung])) if rung else set()
    return not ((finer_t - gold_t) & qt)


def locator_lint(q, span_surface, other_surfaces):
    """Reject locator questions that identify this span through another detected span.

    Rung >= 1 questions are asked over anonymized `out_p`; if the question names another
    sensitive span, the reader cannot ground it after that other span is replaced.
    """
    qt = _tokens(q or "")
    this = canon(span_surface or "")
    for surface in other_surfaces or []:
        if canon(surface or "") == this:
            continue
        st = _tokens(surface or "")
        if st and st <= qt:
            return False
    return True


def validate_ladder(entries, reader_hi_final, reader_hi_p, reader_lo_final, reader_lo_p, th):
    """Per-rung anchor validation with injected readers.

    Rung 0 uses post-inversion anchors (`out_final`), matching the exact echo channel.
    Semantic rungs use pre-inversion anchors (`out_p`), matching the semantic channel.
    An entry survives when the ceiling answer entails its rung and the floor answer does
    not.
    """
    kept, rows = [], []
    for e in entries:
        q = e.get("q", "")
        rung = int(e.get("rung", 0))
        rungs = e.get("rungs") or [e.get("a") or e.get("surface", "")]
        reader_hi = reader_hi_final if rung == 0 else reader_hi_p
        reader_lo = reader_lo_final if rung == 0 else reader_lo_p
        hi_answer = reader_hi(q) or ""
        lo_answer = reader_lo(q) or ""
        hi_score = entail_score(hi_answer, rungs, rung)
        lo_score = entail_score(lo_answer, rungs, rung)
        if hi_score < th:
            verdict = "ceiling"
        elif lo_score >= th:
            verdict = "floor"
        else:
            verdict = "kept"
        row = {
            "id": e.get("id") or e.get("surface"),
            "surface": e.get("surface"),
            "rung": rung,
            "q": q,
            "hi_answer": hi_answer,
            "lo_answer": lo_answer,
            "hi_score": round(hi_score, 3),
            "lo_score": round(lo_score, 3),
            "verdict": verdict,
        }
        rows.append(row)
        out = {**e, "validation": row}
        if verdict == "kept":
            kept.append(out)
    return kept, rows


def mc_shuffle(options, seed_key):
    """Deterministic per-call multiple-choice option shuffle."""
    out = list(options or [])
    seed = int.from_bytes(hashlib.sha256(str(seed_key).encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(out)
    return out


def _decision_span_ids(entry):
    spans = entry.get("detected_spans") or entry.get("spans") or []
    depends = entry.get("depends_on") or []
    dep_text = " ".join(canon(str(d)) for d in depends)
    ids = []
    for i, span in enumerate(spans):
        sid = span.get("id") or span.get("span_id") or str(i)
        surface = canon(str(span.get("surface", ""))).strip()
        if surface and surface in dep_text:
            ids.append(sid)
    return ids


def validate_decisions(entries, reader_mc_hi, reader_mc_lo):
    """Validate multiple-choice decision probes with injected pinned readers."""
    kept, rows = [], []
    for idx, e in enumerate(entries):
        q = e.get("q", "")
        gold = e.get("gold")
        hi_options = mc_shuffle(e.get("options") or [], f"{q}|{idx}|hi")
        lo_options = mc_shuffle(e.get("options") or [], f"{q}|{idx}|lo")
        hi_pick = reader_mc_hi(q, hi_options)
        lo_pick = reader_mc_lo(q, lo_options)
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
            "span_ids": _decision_span_ids(e),
            "verdict": verdict,
        }
        rows.append(row)
        out = {**e, "span_ids": _decision_span_ids(e), "validation": row}
        if verdict == "kept":
            kept.append(out)
    return kept, rows


def _parse_json_list(reply: str) -> list | None:
    if not reply or "<think>" in reply:
        return None
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else None


def sentence_of(doc_text: str, surface: str) -> str:
    """Naive containing-sentence extraction (context line 'appears in' of the prompt)."""
    i = doc_text.lower().find(surface.lower())
    if i < 0:
        return ""
    start = max(doc_text.rfind(c, 0, i) for c in ".!?\n") + 1
    ends = [j for j in (doc_text.find(c, i + len(surface)) for c in ".!?\n") if j >= 0]
    return doc_text[start:min(ends) + 1 if ends else len(doc_text)].strip()


LOCAL_BASE_URL = "http://localhost:8060/v1"


class _AnthropicTeacher:
    """Minimal Anthropic Messages client for proxies that expose claude-* models only via
    anthropic_messages (not openai_chat). Bypasses the CLOAK_LLM_CACHE disk cache —
    acceptable for teacher comparison spikes; the probe caches memoize the parsed result."""

    def __init__(self, model: str, base_url: str):
        self.model, self.base_url = model, base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        import httpx
        r = httpx.post(f"{self.base_url}/messages",
                       headers={"x-api-key": "x", "anthropic-version": "2023-06-01"},
                       json={"model": self.model, "max_tokens": 1024, "temperature": 0.0,
                             "messages": [{"role": "user", "content": prompt}]},
                       timeout=180)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def _teacher(model: str, base_url: str = LOCAL_BASE_URL):
    if model.startswith("claude-") and "localhost" not in base_url:
        return _AnthropicTeacher(model, base_url)
    from cloak.llm import LLMClient
    # chat_template_kwargs is llama.cpp-specific; remote OpenAI-compatible providers may
    # reject unknown extra_body keys, so it is sent only to the local proxy
    is_local = "localhost" in base_url
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if is_local else None
    # OpenRouter (hosted teacher, same wiring as the lattice producer) authenticates with
    # OPENROUTER_API_KEY; local proxy is keyless.
    import os
    api_key = (os.environ.get("OPENROUTER_API_KEY") if "openrouter.ai" in base_url else "x") or "x"
    return LLMClient(model, base_url=base_url, api_key=api_key,
                     temperature=0.0, max_tokens=1024, extra_body=extra)


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path and path.exists() else {}


def _covered(entries: list[dict], model: str, pv: int) -> set[str]:
    return {e["surface"] for e in entries if e.get("teacher") == model and e.get("pv") == pv}


def ladder_probes_for_docs(docs: list[dict], spans_of: dict, corpus: str, workers: int = 6,
                           model: str = TEACHER_MODEL, base_url: str = LOCAL_BASE_URL,
                           cache_path: Path = LADDER_CACHE) -> dict:
    """docs: corpora rows; spans_of: doc_id -> env spans (each {surface, type, actions}).
    Returns {doc_id: [{"surface", "rung", "q", "a"}...]} for lattice-bearing spans; teacher
    fills cache misses. Lint-rejected rungs are simply absent (re-asked only on pv bump)."""
    from cloak.concurrent import pmap

    kind = OUTPUT_KIND.get(corpus, "summary")
    cache = _load(cache_path)
    todo = []
    for d in docs:
        have = _covered(cache.get(d["id"], []), model, LADDER_PV)
        all_surfaces = [s.get("surface", "") for s in spans_of.get(d["id"], [])]
        for s in spans_of.get(d["id"], []):
            levels = span_levels(s)
            if not levels or s["surface"] in have:
                continue
            rungs = rung_phrases(s["surface"], levels)
            todo.append({"doc_id": d["id"], "surface": s["surface"], "type": s.get("type", ""),
                         "other_surfaces": [x for x in all_surfaces if canon(x) != canon(s["surface"])],
                         "rungs": rungs,
                         "prompt": LADDER_PROMPT.format(
                             output_kind=kind, doc=d["text"], surface=s["surface"],
                             type=s.get("type", ""),
                             sentence=s.get("sent") or sentence_of(d["text"], s["surface"]),
                             rungs="\n".join(f"  {i}: {r}" for i, r in enumerate(rungs)))})
    if todo:
        teacher = _teacher(model, base_url)
        replies = pmap(lambda t: teacher.generate(t["prompt"]), todo, workers=workers)
        n_bad = 0
        for t, r in zip(todo, replies):
            rows = _parse_json_list((r or "").strip())
            if rows is None:
                n_bad += 1
                continue
            for row in rows:
                rung = row.get("rung")
                if (isinstance(rung, int) and 0 <= rung < len(t["rungs"])
                        and not _empty_gold(t["rungs"][rung])
                        and lint_rung(row.get("q", ""), t["rungs"], rung)
                        and (rung == 0 or locator_lint(row.get("q", ""), t["surface"],
                                                       t["other_surfaces"]))):
                    cache.setdefault(t["doc_id"], []).append(
                        {"surface": t["surface"], "rung": rung, "q": row["q"].strip(),
                         "a": t["rungs"][rung], "rungs": t["rungs"],
                         "teacher": model, "pv": LADDER_PV})
        if n_bad:
            print(f"ladder_probes: {n_bad}/{len(todo)} teacher replies unparseable", flush=True)
        if cache_path:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    return {d["id"]: [e for e in cache.get(d["id"], [])
                      if e.get("teacher") == model and e.get("pv") == LADDER_PV]
            for d in docs}


def decision_probes_for_docs(docs: list[dict], out_hi_of: dict, corpus: str, k: int = 4,
                             workers: int = 6, model: str = TEACHER_MODEL,
                             base_url: str = LOCAL_BASE_URL,
                             cache_path: Path = DECISION_CACHE) -> dict:
    """docs: corpora rows; out_hi_of: doc_id -> ceiling output. One teacher call per doc.
    Structural validation only (gold in 3-5 options, question form); reader-side ceiling/floor
    validation happens at probe-build time, not here."""
    from cloak.concurrent import pmap

    if corpus not in DECISION_KINDS:
        return {d["id"]: [] for d in docs}
    kind = OUTPUT_KIND.get(corpus, "summary")
    cache = _load(cache_path)
    todo = [d for d in docs
            if d["id"] in out_hi_of
            and not any(e.get("teacher") == model and e.get("pv") == DECISION_PV
                        for e in cache.get(d["id"], []))]
    if todo:
        teacher = _teacher(model, base_url)
        replies = pmap(lambda d: teacher.generate(DECISION_PROMPT.format(
            output_kind=kind, k=k, decision_kinds=DECISION_KINDS[corpus],
            doc=d["text"], out_hi=out_hi_of[d["id"]])), todo, workers=workers)
        n_bad = 0
        for d, r in zip(todo, replies):
            rows = _parse_json_list((r or "").strip())
            if rows is None:
                n_bad += 1
                continue
            kept = []
            for row in rows[:k]:
                q, opts, gold = row.get("q", "").strip(), row.get("options"), row.get("gold")
                if (q.endswith("?") and isinstance(opts, list) and 3 <= len(opts) <= 5
                        and gold in opts):
                    kept.append({"q": q, "options": opts, "gold": gold,
                                 "depends_on": row.get("depends_on") or [],
                                 "teacher": model, "pv": DECISION_PV})
            cache[d["id"]] = cache.get(d["id"], []) + kept
        if n_bad:
            print(f"decision_probes: {n_bad}/{len(todo)} teacher replies unparseable",
                  flush=True)
        if cache_path:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    return {d["id"]: [e for e in cache.get(d["id"], [])
                      if e.get("teacher") == model and e.get("pv") == DECISION_PV]
            for d in docs}


if __name__ == "__main__":
    rungs = rung_phrases("hypothyroidism", ["an endocrine condition", "a chronic condition"])
    assert rungs[0] == "hypothyroidism" and len(rungs) == 3
    assert span_levels({
        "surface": "heart failure",
        "actions": [
            {"mode": "level", "fill": "a physical condition", "aset": 1313},
            {"mode": "level", "fill": "a cardiovascular disease", "aset": 28},
            {"mode": "level", "fill": "heart failure", "aset": 1.0},
            {"mode": "placeholder", "fill": "<HEALTH_CONDITION_1>"},
        ],
    }) == ["a cardiovascular disease", "a physical condition"]
    # entailment: a finer answer satisfies a coarser rung
    assert entail_score("hypothyroidism", rungs, 1) == 1.0
    assert entail_score("an endocrine condition", rungs, 2) == 1.0
    assert entail_score("no idea", rungs, 2) == 0.0
    # lint: generic token shared with own gold is fine; finer-rung distinctive token leaks
    assert lint_rung("What body-system category of condition is managed with medication?",
                     rungs, 1)
    assert not lint_rung("Is the condition endocrine?", rungs, 1)          # yes/no
    assert not lint_rung("Which endocrine condition does the patient have?", rungs, 1)  # gold
    assert not lint_rung("What kind of issue is the hypothyroidism?", rungs, 2)  # finer leak
    assert lint_rung("What kind of ongoing health issue does the patient have?", rungs, 2)
    assert _parse_json_list('noise [{"rung": 0, "q": "x?", "a": "y"}] tail') is not None
    assert _parse_json_list("<think>...</think>[]") is None
    assert _empty_gold("something") and _empty_gold("A physical condition.")
    assert not _empty_gold("a cardiovascular disease") and not _empty_gold("an endocrine condition")
    s = sentence_of("He said hi. She takes Synthroid daily. End.", "synthroid")
    assert s == "She takes Synthroid daily.", s
    print("ladder_probes.py self-check OK")
