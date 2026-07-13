"""Round-trip reward (spec docs/specs/RL/roundtrip-ranker-infiller.md, Phase 1).

R_rt = realized fact recall (graded mean token-F1) on out_final over a doc's train-split
probes, where out_final = invert(Remote(task_prompt(doc_p)), R). Deterministic given doc_p:
pinned model, temperature 0, single-flight generation, and content-addressed disk cache
(CLOAK_LLM_CACHE) — the determinism is load-bearing (cache = reward memoization = ExIt
pool; spec §Determinism under concurrency).

THE reward pin (changing any re-gates): RT_MODEL = "gemma 4 (E4B)" served at
RT_BASE_URL = "http://localhost:8060/v1", temperature 0, max_tokens 1024, non-thinking.
The extractor is part of the reward pin: legacy rewards are pinned to the `invert` cascade,
frozen-extractor rewards are keyed by `extractor_version`, and cached rewards are valid only
under the pin they were produced with.
"""
import os

from cloak.extract import invert
from cloak.tasks import SCHEMA_TEMPLATE, TASK_TEMPLATE
from cloak.train.qa_builder import read_context_batch, score_utility
from cloak.train.ladder_probes import entail_score, mc_shuffle
from cloak.train.reward import (_max_by_fact, _read_batch, _read_mc_batch, canon,
                                decision_prompt, fact_f1s, fact_score, mc_score,
                                W_EXACT, W_SEM)
from cloak.train.schema_task import schema_field_score

RT_MODEL = "gemma 4 (E4B)"   # THE pin (spec components table); changing it re-gates.
RT_BASE_URL = "http://localhost:8060/v1"   # THE endpoint pin; part of the reward pin.
# User decision 2026-07-05 (results/thinking_mode_probe.json): gemma honors
# enable_thinking:false (clean non-thinking output, all probe facts restated in ~150 tok);
# LFM2.5-8B-A1B cannot disable thinking (the flag leaks <think> in-band, truncating at
# this budget) and moved to the probe-teacher role instead.
MAX_TOKENS = 1024   # raised from 512 (2026-07-05, pre-gate calibration): full ACI notes hit
                    # the 512 cap mid-sentence (measured: out_len ~532 tok, tail truncated),
                    # killing ceiling-anchor validation on facts from later note sections.
                    # gemma finishes real notes in ~400-700 tok; 1024 is headroom, not a target.

_client = None


def _remote():
    global _client
    if _client is None:
        from cloak.llm import LLMClient
        assert os.getenv("CLOAK_LLM_CACHE"), \
            "round-trip reward requires CLOAK_LLM_CACHE (determinism + cost)"
        _client = LLMClient(RT_MODEL, base_url=RT_BASE_URL, temperature=0.0,
                            max_tokens=MAX_TOKENS,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                            single_flight=True)
    return _client


def _kept(entries: list[dict] | None) -> list[dict]:
    return [e for e in (entries or []) if e.get("kept", True) is not False]


def _ladder_key(entry: dict, fallback: int) -> str:
    return str(entry.get("span_id") or canon(entry.get("surface", "")) or entry.get("id")
               or fallback)


def _rung_question(entry: dict) -> str | None:
    return entry.get("question") or entry.get("q")


def _score_ladder(ladder: list[dict], out_final: str, out_p: str,
                  refresh: bool) -> tuple[list[float], list[float]]:
    """Return (per-span carrier parts, raw rung-0 echo F1s)."""
    entries = _kept(ladder)
    groups: dict[str, dict[str, list[float]]] = {}
    exact_rows, sem_rows = [], []
    for i, entry in enumerate(entries):
        q = _rung_question(entry)
        rungs = entry.get("rungs") or [entry.get("surface") or entry.get("a") or ""]
        if not q or not rungs:
            continue
        rung = int(entry.get("rung", 0))
        aliases = entry.get("aliases") or []
        key = _ladder_key(entry, i)
        groups.setdefault(key, {"exact": [], "sem": []})
        if rung == 0:
            exact_rows.append((key, q, entry.get("surface") or rungs[0], aliases))
        elif rung >= 1:
            sem_rows.append((key, q, rungs, rung, aliases))

    echo_f1s = []
    if exact_rows:
        answers = _read_batch([q for _, q, _, _ in exact_rows], out_final, refresh=refresh)
        for answer, (key, _q, surface, aliases) in zip(answers, exact_rows):
            # accept a surface-equivalent alias the note may have used (HTN vs hypertension),
            # matching the acceptance rule the probe was validated under (ladder_probes)
            score = max(fact_score(answer, s) for s in [surface, *aliases])
            echo_f1s.append(score)
            groups[key]["exact"].append(score)

    if sem_rows:
        answers = _read_batch([q for _, q, _, _, _ in sem_rows], out_p, refresh=refresh)
        for answer, (key, _q, rungs, rung, aliases) in zip(answers, sem_rows):
            groups[key]["sem"].append(entail_score(answer, rungs, rung, aliases))

    parts = []
    for scores in groups.values():
        exact = sum(scores["exact"]) / len(scores["exact"]) if scores["exact"] else 0.0
        sem = sum(scores["sem"]) / len(scores["sem"]) if scores["sem"] else 0.0
        parts.append(W_EXACT * exact + W_SEM * sem)
    return parts, echo_f1s


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


def _template(job: dict) -> str:
    wants_schema = job.get("template") == "schema" or job.get("schema")
    if wants_schema and job["corpus"] in SCHEMA_TEMPLATE:
        return SCHEMA_TEMPLATE[job["corpus"]]
    return TASK_TEMPLATE[job["corpus"]]


def _carrier_enabled(job: dict) -> bool:
    return bool(job.get("ladder") or job.get("decisions") or job.get("schema"))


def roundtrip_batch(
    jobs: list[dict],
    workers: int = 6,
    extractor_models: dict | None = None,
    reader_refresh: bool = False,
) -> list[dict]:
    """jobs: [{corpus, doc_p, R, probes, ladder?, decisions?, out_hi?, schema?}].
    Returns [{out_p, out_final, f1s, recall, ...}].
    recall = deployed fact_recall (per-fact max, mean over facts), None when no probes.
    With carrier fields, recall is R_carrier's unweighted mean over available components.
    f1s stays the raw per-question list (the support scan counts per-question flip deltas).

    Each job's full gen->invert->read->score runs on one worker, so up to `workers` jobs run
    concurrently (across-context parallelism -> the served `-np 6` slots) while `fact_f1s`
    stays serial WITHIN a job (its questions share out_final -> note-prefix KV reuse). workers
    defaults to 6 to match the served slot count; gen is single-flight/cached while
    reader_refresh can bypass the reader cache for winner re-verification (spec
    §Determinism under concurrency)."""
    from cloak.concurrent import pmap
    remote = _remote()
    if extractor_models is not None:
        from cloak import frozen_extractor

    def _one(j):
        op = remote.generate(_template(j).format(doc=j["doc_p"]))
        if extractor_models is None:
            out_final, _ = invert(op, j["R"])
            extractor_version = None
        else:
            out_final, _ = frozen_extractor.extract(
                j.get("doc_p"),
                j["R"],
                op,
                models=extractor_models,
            )
            extractor_version = frozen_extractor.extractor_version()
        if j.get("utility_artifact") is not None:
            scored = score_utility(
                j["utility_artifact"],
                j["doc_id"],
                doc_p=j["doc_p"],
                out_final=out_final,
                reader=read_context_batch,
                reader_refresh=reader_refresh,
            )
            result = {
                "out_p": op,
                "out_final": out_final,
                "f1s": [],
                "recall": scored["utility"],
                "component_scores": scored["component_scores"],
            }
            if extractor_version is not None:
                result["extractor_version"] = extractor_version
            return result
        if _carrier_enabled(j):
            span_parts, f1s = _score_ladder(j.get("ladder") or [], out_final, op,
                                            reader_refresh)
            components = []
            span_score = None
            if span_parts:
                span_score = sum(span_parts) / len(span_parts)
                components.append(span_score)
            decision_score = _score_decisions(j.get("decisions") or [], op,
                                              reader_refresh)
            if decision_score is not None:
                components.append(decision_score)
            schema_score = None
            if j.get("schema") and j.get("out_hi"):
                schema_score = schema_field_score(out_final, j["out_hi"])
                if schema_score is not None:
                    components.append(schema_score)
            result = {
                "out_p": op,
                "out_final": out_final,
                "f1s": f1s,
                "recall": (sum(components) / len(components)) if components else None,
                "span_parts": span_parts,
                "span_score": span_score,
                "decision_score": decision_score,
                "schema_score": schema_score,
            }
            if extractor_version is not None:
                result["extractor_version"] = extractor_version
            return result
        f1s = fact_f1s(out_final, j["probes"], refresh=reader_refresh)
        by_fact = _max_by_fact(j["probes"], f1s)
        result = {"out_p": op, "out_final": out_final, "f1s": f1s,
                  "recall": (sum(by_fact.values()) / len(by_fact)) if by_fact else None}
        if extractor_version is not None:
            result["extractor_version"] = extractor_version
        return result

    return pmap(_one, jobs, workers=workers)


if __name__ == "__main__":
    # LIVE smoke (hits the proxy once; requires CLOAK_LLM_CACHE and the proxy up):
    #   CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -m cloak.train.roundtrip
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
