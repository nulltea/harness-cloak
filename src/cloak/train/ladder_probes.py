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
_EMPTY_GOLDS = {
    "something",
    "a thing",
    "an issue",
    "a disorder",
    "a condition",
    "a physical condition",
    "a problem",
}


def _empty_gold(phrase: str) -> bool:
    f = re.sub(r"\s+", " ", canon(phrase)).strip().rstrip(".")
    return f in _EMPTY_GOLDS or is_type_name_phrase(phrase)


LADDER_CACHE = Path("data/ladder_probes.json")
DECISION_CACHE = Path("data/decision_probes.json")
# pv2: clinical-role grounding, positional/enumeration forbidden.
# pv3: JSON-object output ({"probes": [...]}) enforced via response_format=json_object.
# pv4: two questions per span — rung 0 (exact) + rung 1 (finest generalization) — instead of
#      one per rung. On nested lattices no question selects one level (measured: identical
#      question repeated across rungs), so per-rung questions were one measurement scored at
#      L thresholds, and monotone acceptance made coarsening nearly free.
LADDER_PV = 4
# pv2: JSON-object output ({"decisions": [...]}) enforced via response_format=json_object.
# pv3: decisions must turn on a fact's identity/category (placeholder-breakable), not plan
#      readback or outside-knowledge appropriateness; DECISION_KINDS retargeted to category.
# pv4: the question must NOT name the target fact (world-knowledge-trivia shape measured in
#      the 2026-07-12 super sweep: every decision named its fact and asked a generic
#      property); identify the situation via documented circumstances. Enforced by
#      lint_decision at generation time.
DECISION_PV = 4

OUTPUT_KIND = {
    "aci": "clinical note",
    "mts": "clinical note",
    "clinical": "clinical note",
    "lexsum": "case summary",
    "wikibio": "biography summary",
    "enron": "email reply",
    "aeslc": "email subject line",
    "qmsum": "meeting summary",
}

DECISION_KINDS = {
    "clinical": (
        "which medical specialist should treat or follow up a problem, which body-system "
        "category a condition belongs to, or whether a class of medication or treatment is "
        "indicated"
    ),
    "lexsum": (
        "which area of law governs a claim, likely prevailing party by claim type, or which "
        "category of remedy applies"
    ),
}
DECISION_KINDS["aci"] = DECISION_KINDS["mts"] = DECISION_KINDS["clinical"]

# TODO: this prompt as written is clinical-specific (meantions "clinical consequence") - when expending to other datasets - this must be fixed!
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

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "and",
    "or",
    "with",
    "by",
}
_YESNO = re.compile(
    r"^(is|are|was|were|does|did|do|has|have|had|can|could|should|would|will"
    r"|may|might)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", canon(text)) if w not in _STOP}


def rung_phrases(surface: str, levels: list[str]) -> list[str]:
    """rungs[0] = exact surface, rungs[l] = l-th lattice level (specific -> broad)."""
    return [surface, *levels]


def span_levels(span: dict) -> list[str]:
    """Ladder levels for a span, specific -> broad, from lattice_profiles.json — the single
    source of truth (its `levels` is validated monotone and never contains the surface, so no
    sort or keep-drop is needed). Keyed by (surface, runtime_type); aliases resolve in the
    loader. Falls back to the span's own baked level actions ONLY when the profile has no
    entry, so legacy env artifacts (coarse types with no profile row) still work.
    """
    from cloak.lattice_profiles import lookup_levels

    levels = lookup_levels(span.get("surface", ""), span.get("type", ""))
    if levels:
        return list(levels)
    # legacy fallback: env-baked action fills (older envs predate the profile / use coarse types)
    surface = canon(span.get("surface", ""))
    acts = [
        a
        for a in span.get("actions", [])
        if a.get("mode") == "level" and canon(a.get("fill") or "") != surface
    ]
    acts.sort(key=lambda a: a.get("aset") or 0)
    return [a["fill"] for a in acts]


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


def validate_ladder(
    entries, reader_hi_final, reader_hi_p, reader_lo_final, reader_lo_p, th
):
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
        aliases = e.get("aliases") or []
        reader_hi = reader_hi_final if rung == 0 else reader_hi_p
        reader_lo = reader_lo_final if rung == 0 else reader_lo_p
        hi_answer = reader_hi(q) or ""
        lo_answer = reader_lo(q) or ""
        hi_score = entail_score(hi_answer, rungs, rung, aliases)
        lo_score = entail_score(lo_answer, rungs, rung, aliases)
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
    seed = int.from_bytes(
        hashlib.sha256(str(seed_key).encode("utf-8")).digest()[:8], "big"
    )
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


def _parse_json_list(reply: str) -> list | None:
    """Extract the probe list from a teacher reply. Accepts the enforced JSON-object form
    ({"probes"/"decisions": [...]}, or any single list-valued key) and, as a fallback for
    unconstrained providers, a bare JSON list."""
    if not reply or "<think>" in reply:
        return None
    obj = re.search(r"\{.*\}", reply, re.DOTALL)
    if obj:
        try:
            v = json.loads(obj.group())
        except json.JSONDecodeError:
            v = None
        if isinstance(v, dict):
            lists = [x for x in v.values() if isinstance(x, list)]
            if len(lists) == 1:
                return lists[0]
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
    return doc_text[start : min(ends) + 1 if ends else len(doc_text)].strip()


LOCAL_BASE_URL = "http://localhost:8060/v1"


class _AnthropicTeacher:
    """Minimal Anthropic Messages client for proxies that expose claude-* models only via
    anthropic_messages (not openai_chat). Bypasses the CLOAK_LLM_CACHE disk cache —
    acceptable for teacher comparison spikes; the probe caches memoize the parsed result."""

    def __init__(self, model: str, base_url: str):
        self.model, self.base_url = model, base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        import httpx

        r = httpx.post(
            f"{self.base_url}/messages",
            headers={"x-api-key": "x", "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def _teacher(model: str, base_url: str = LOCAL_BASE_URL):
    if model.startswith("claude-") and "localhost" not in base_url:
        return _AnthropicTeacher(model, base_url)
    import os
    from cloak.llm import LLMClient

    if "openrouter.ai" in base_url:
        # Hosted teacher (Nemotron etc. are reasoning models). Authenticate with
        # OPENROUTER_API_KEY; response_format hard-enforces a JSON object; reasoning=exclude asks
        # the provider not to return reasoning. But the :free endpoint load-balances across
        # providers and some IGNORE exclude, dumping 30-37 KB of reasoning into `content` — and a
        # reasoning model burns any max_tokens cap on that reasoning before emitting the JSON
        # (measured: with an 8000 cap, ultra truncated mid-reasoning -> unparseable). So set NO
        # max_tokens cap: let reasoning run to completion so the JSON is always emitted at the end
        # (_parse_json_list recovers the trailing object/list even after a reasoning preamble).
        api_key = os.environ.get("OPENROUTER_API_KEY") or "x"
        return LLMClient(
            model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            response_format={"type": "json_object"},   # hard-enforce a JSON object reply
            extra_body={"reasoning": {"exclude": True}},
        )
    # chat_template_kwargs is llama.cpp-specific; sent only to the local proxy.
    extra = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if "localhost" in base_url
        else None
    )
    return LLMClient(
        model,
        base_url=base_url,
        api_key="x",
        temperature=0.0,
        max_tokens=1024,
        extra_body=extra,
    )


def _safe_generate(teacher, prompt: str) -> str:
    """Teacher call that never aborts the pmap batch. The client already retries 429/5xx with
    backoff (cloak.llm max_retries); if a call still exhausts retries, return '' so the batch
    finishes — an empty reply is counted unparseable and re-asked next run (successful calls
    are cached per-call, so re-runs resume)."""
    try:
        return teacher.generate(prompt)
    except Exception as e:  # noqa: BLE001 — degrade one failed call, not the whole batch
        print(
            f"ladder_probes: teacher call failed ({type(e).__name__}: {e}); "
            f"treating as unparseable",
            flush=True,
        )
        return ""


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path and path.exists() else {}


def _reusable(entry: dict, model: str, want: dict) -> bool:
    """A cached ladder entry is reusable only if it was built by THIS teacher+pv AND its rungs
    still match the current profile-sourced ladder for its surface (want: canon(surface) ->
    rungs, built from THIS run's detected spans). This is the guard against cross-run / cross-path
    cache leakage: a surface not detected this run (a prior env-path 'dragon'), or a detected
    surface whose lattice changed since it was cached, is not reusable and is re-generated."""
    if entry.get("teacher") != model or entry.get("pv") != LADDER_PV:
        return False
    key = canon(entry.get("surface", ""))
    return key in want and list(entry.get("rungs") or []) == list(want[key])


def ladder_probes_for_docs(
    docs: list[dict],
    spans_of: dict,
    corpus: str,
    workers: int = 6,
    model: str = TEACHER_MODEL,
    base_url: str = LOCAL_BASE_URL,
    cache_path: Path = LADDER_CACHE,
    all_surfaces_of: dict | None = None,
    reject_sink: list | None = None,
    gen_sink: list | None = None,
) -> dict:
    """docs: corpora rows; spans_of: doc_id -> spans (each {surface, type, ...}).
    Returns {doc_id: [{"surface", "rung", "q", "a"}...]} for lattice-bearing spans; teacher
    fills cache misses.

    all_surfaces_of: doc_id -> every detected surface (lattice + placeholder + quasi); the
    hidden-detail list the prompt tells the teacher to avoid and the locator-lint checks
    against. Falls back to spans_of surfaces when None.
    reject_sink: if given, generation-stage drops append
    {doc_id, surface, rung, q, gate, gold} so callers can report why a rung was cut
    (gates: parse, bad_rung, empty_gold, lint, locator)."""
    from cloak.concurrent import pmap

    from cloak.lattice_profiles import lookup_aliases

    kind = OUTPUT_KIND.get(corpus, "summary")
    cache = _load(cache_path)
    # want[doc_id][canon(surface)] = current profile-sourced rungs for each detected lattice span.
    # Cache reuse and the return are both scoped to this — never to (teacher, pv) alone.
    want_of = {
        d["id"]: {
            canon(s["surface"]): rung_phrases(s["surface"], span_levels(s))
            for s in spans_of.get(d["id"], [])
            if span_levels(s)
        }
        for d in docs
    }
    todo = []
    for d in docs:
        want = want_of[d["id"]]
        doc_cache = cache.get(d["id"], [])
        hidden = (all_surfaces_of or {}).get(d["id"])
        if hidden is None:
            hidden = [s.get("surface", "") for s in spans_of.get(d["id"], [])]
        for s in spans_of.get(d["id"], []):
            key = canon(s["surface"])
            if key not in want:  # no profile levels -> not a probe span
                continue
            if any(
                _reusable(e, model, want) and canon(e.get("surface", "")) == key
                for e in doc_cache
            ):  # valid cached entry already -> skip re-generation
                continue
            rungs = want[key]
            other = [x for x in hidden if x and canon(x) != canon(s["surface"])]
            aliases = lookup_aliases(s["surface"], s.get("type", ""))
            todo.append(
                {
                    "doc_id": d["id"],
                    "surface": s["surface"],
                    "type": s.get("type", ""),
                    "other_surfaces": other,
                    "rungs": rungs,
                    "aliases": aliases,
                    "prompt": LADDER_PROMPT.format(
                        output_kind=kind,
                        doc=d["text"],
                        surface=s["surface"],
                        type=s.get("type", ""),
                        sentence=s.get("sent") or sentence_of(d["text"], s["surface"]),
                        rung1=rungs[1],
                    ),
                }
            )
    if todo:
        teacher = _teacher(model, base_url)
        replies = pmap(
            lambda t: _safe_generate(teacher, t["prompt"]), todo, workers=workers
        )
        n_bad = 0

        def _rej(t, rung, q, gate, gold=""):
            if reject_sink is not None:
                reject_sink.append(
                    {
                        "doc_id": t["doc_id"],
                        "surface": t["surface"],
                        "rung": rung,
                        "q": q,
                        "gate": gate,
                        "gold": gold,
                    }
                )

        for t, r in zip(todo, replies):
            if gen_sink is not None:
                gen_sink.append(
                    {
                        "doc_id": t["doc_id"],
                        "surface": t["surface"],
                        "type": t["type"],
                        "rungs": t["rungs"],
                        "raw": r,
                    }
                )
            rows = _parse_json_list((r or "").strip())
            if rows is None:
                n_bad += 1
                _rej(t, None, "", "parse")
                continue
            for row in rows:
                rung, q = row.get("rung"), (row.get("q") or "").strip()
                if not (isinstance(rung, int) and rung in (0, 1)):
                    _rej(t, rung, q, "bad_rung")
                    continue
                gold = t["rungs"][rung]
                if _empty_gold(gold):
                    _rej(t, rung, q, "empty_gold", gold)
                elif not lint_rung(q, t["rungs"], rung):
                    _rej(t, rung, q, "lint", gold)
                elif rung != 0 and not locator_lint(
                    q, t["surface"], t["other_surfaces"]
                ):
                    _rej(t, rung, q, "locator", gold)
                else:
                    cache.setdefault(t["doc_id"], []).append(
                        {
                            "surface": t["surface"],
                            "rung": rung,
                            "q": q,
                            "a": gold,
                            "rungs": t["rungs"],
                            "aliases": t["aliases"],
                            "teacher": model,
                            "pv": LADDER_PV,
                        }
                    )
        if n_bad:
            print(
                f"ladder_probes: {n_bad}/{len(todo)} teacher replies unparseable",
                flush=True,
            )
        if cache_path:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    # Return ONLY entries that match this run's detected spans + current lattice (want_of),
    # so stale cross-run / cross-path cache entries never leak into the artifact.
    return {
        d["id"]: [
            e for e in cache.get(d["id"], []) if _reusable(e, model, want_of[d["id"]])
        ]
        for d in docs
    }


def decision_probes_for_docs(
    docs: list[dict],
    out_hi_of: dict,
    corpus: str,
    k: int = 4,
    workers: int = 6,
    model: str = TEACHER_MODEL,
    base_url: str = LOCAL_BASE_URL,
    cache_path: Path = DECISION_CACHE,
    lattice_surfaces_of: dict | None = None,
    gen_sink: list | None = None,
) -> dict:
    """docs: corpora rows; out_hi_of: doc_id -> ceiling output. One teacher call per doc.
    Structural validation only (gold in 3-5 options, question form); reader-side ceiling/floor
    validation happens at probe-build time, not here.

    lattice_surfaces_of: doc_id -> detected lattice surfaces; questions naming one are dropped
    by lint_decision."""
    from cloak.concurrent import pmap

    if corpus not in DECISION_KINDS:
        return {d["id"]: [] for d in docs}
    kind = OUTPUT_KIND.get(corpus, "summary")
    cache = _load(cache_path)
    todo = [
        d
        for d in docs
        if d["id"] in out_hi_of
        and not any(
            e.get("teacher") == model and e.get("pv") == DECISION_PV
            for e in cache.get(d["id"], [])
        )
    ]
    if todo:
        teacher = _teacher(model, base_url)
        replies = pmap(
            lambda d: _safe_generate(
                teacher,
                DECISION_PROMPT.format(
                    output_kind=kind,
                    k=k,
                    decision_kinds=DECISION_KINDS[corpus],
                    doc=d["text"],
                    out_hi=out_hi_of[d["id"]],
                ),
            ),
            todo,
            workers=workers,
        )
        n_bad = 0
        for d, r in zip(todo, replies):
            if gen_sink is not None:
                gen_sink.append({"doc_id": d["id"], "raw": r})
            rows = _parse_json_list((r or "").strip())
            if rows is None:
                n_bad += 1
                continue
            kept = []
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
                    kept.append(
                        {
                            "q": q,
                            "options": opts,
                            "gold": gold,
                            "depends_on": row.get("depends_on") or [],
                            "teacher": model,
                            "pv": DECISION_PV,
                        }
                    )
            cache[d["id"]] = cache.get(d["id"], []) + kept
        if n_bad:
            print(
                f"decision_probes: {n_bad}/{len(todo)} teacher replies unparseable",
                flush=True,
            )
        if cache_path:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    return {
        d["id"]: [
            e
            for e in cache.get(d["id"], [])
            if e.get("teacher") == model and e.get("pv") == DECISION_PV
        ]
        for d in docs
    }


if __name__ == "__main__":
    rungs = rung_phrases(
        "hypothyroidism", ["an endocrine condition", "a chronic condition"]
    )
    assert rungs[0] == "hypothyroidism" and len(rungs) == 3
    assert span_levels(
        {
            "surface": "heart failure",
            "actions": [
                {"mode": "level", "fill": "a physical condition", "aset": 1313},
                {"mode": "level", "fill": "a cardiovascular disease", "aset": 28},
                {"mode": "level", "fill": "heart failure", "aset": 1.0},
                {"mode": "placeholder", "fill": "<HEALTH_CONDITION_1>"},
            ],
        }
    ) == ["a cardiovascular disease", "a physical condition"]
    # entailment: a finer answer satisfies a coarser rung
    assert entail_score("hypothyroidism", rungs, 1) == 1.0
    assert entail_score("an endocrine condition", rungs, 2) == 1.0
    assert entail_score("no idea", rungs, 2) == 0.0
    # alias acceptance: a surface-equivalent synonym the note used satisfies rung 0
    assert entail_score("HTN", ["hypertension"], 0) == 0.0  # surface-only: miss
    assert (
        entail_score("HTN", ["hypertension"], 0, ["htn", "high bp"]) == 1.0
    )  # +aliases: hit
    # binary semantic match: sibling category sharing a head noun scores 0, never F1 0.5
    assert entail_score("vascular disease", ["hypertension", "artery disease"], 1) == 0.0
    assert entail_score("coronary artery disease", ["hypertension", "artery disease"], 1) == 1.0
    # lint: generic token shared with own gold is fine; finer-rung distinctive token leaks
    assert lint_rung(
        "What body-system category of condition is managed with medication?", rungs, 1
    )
    assert not lint_rung("Is the condition endocrine?", rungs, 1)  # yes/no
    assert not lint_rung(
        "Which endocrine condition does the patient have?", rungs, 1
    )  # gold
    assert not lint_rung(
        "What kind of issue is the hypothyroidism?", rungs, 2
    )  # finer leak
    assert lint_rung(
        "What kind of ongoing health issue does the patient have?", rungs, 2
    )
    assert not lint_decision("Which body system does a mammogram evaluate?", ["mammogram"])
    assert lint_decision("Which specialist should follow up the screened condition?",
                         ["mammogram"])

    class _Boom:
        def generate(self, _p):
            raise RuntimeError("429 rate limit")

    assert _safe_generate(_Boom(), "x") == ""  # a failed call degrades, never aborts
    assert (
        _safe_generate(type("T", (), {"generate": staticmethod(lambda p: "ok")})(), "x")
        == "ok"
    )
    assert _parse_json_list('noise [{"rung": 0, "q": "x?", "a": "y"}] tail') is not None
    assert _parse_json_list("<think>...</think>[]") is None
    # enforced JSON-object form ({"probes"/"decisions": [...]}) unwraps to the list
    assert _parse_json_list('{"probes": [{"rung": 0, "q": "x?", "a": "y"}]}') == \
        [{"rung": 0, "q": "x?", "a": "y"}]
    assert _parse_json_list('{"decisions": [{"q": "x?", "gold": "y"}]}') == [{"q": "x?", "gold": "y"}]
    assert _empty_gold("something") and _empty_gold("A physical condition.")
    assert not _empty_gold("a cardiovascular disease") and not _empty_gold(
        "an endocrine condition"
    )
    s = sentence_of("He said hi. She takes Synthroid daily. End.", "synthroid")
    assert s == "She takes Synthroid daily.", s
    print("ladder_probes.py self-check OK")
