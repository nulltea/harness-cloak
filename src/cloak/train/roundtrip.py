"""Round-trip reward (spec docs/specs/RL/roundtrip-ranker-infiller.md, Phase 1).

R_rt = realized fact recall (graded mean token-F1) on out_final over a doc's train-split
probes, where out_final = invert(Remote(task_prompt(doc_p)), R). Deterministic given doc_p:
pinned model, temperature 0, single-flight generation, and content-addressed disk cache
(CLOAK_LLM_CACHE) — the determinism is load-bearing (cache = reward memoization = ExIt
pool; spec §Determinism under concurrency).

THE reward pin (changing any re-gates): RT_MODEL = "medgemma-4b-it" served at
RT_BASE_URL = "http://localhost:8060/v1", temperature 0, max_tokens 1024, non-thinking.
The extractor is part of the reward pin: legacy rewards are pinned to the `invert` cascade,
frozen-extractor rewards are keyed by `extractor_version`, and cached rewards are valid only
under the pin they were produced with.
"""
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from cloak.extract import invert, invert_implementation_pin
from cloak.tasks import SCHEMA_TEMPLATE, TASK_TEMPLATE
from cloak.train.interactive_ranker import assemble_action_vector
from cloak.train.qa_builder import (
    UTILITY_SCORER_VERSION,
    finalize_utility_scoring,
    prepare_utility_scoring,
    read_context_batch,
    read_context_set_batch,
    score_utility,
    validate_utility_reader_pin,
)
from cloak.train.ladder_probes import entail_score, mc_shuffle
from cloak.train.reward import (_max_by_fact, _read_batch, _read_mc_batch, canon,
                                decision_prompt, fact_f1s, fact_score, mc_score,
                                W_EXACT, W_SEM)
from cloak.train.schema_task import schema_field_score
from cloak.train.utility_cache import (
    UtilityCache,
    UtilityRequest,
    UtilityResult,
    make_result,
    stable_hash,
    utility_binding,
)

# User re-pin 2026-07-23: medgemma-4b-it replaces gemma 4 (E4B) as the remote model —
# one clinical-tuned model serves both rewrite and reader phases; utility cache was
# empty at switch time, so no cached results were invalidated.
RT_MODEL = "medgemma-4b-it"   # THE pin (spec components table); changing it re-gates.
RT_BASE_URL = "http://localhost:8060/v1"   # THE endpoint pin; part of the reward pin.
# User decision 2026-07-05 (results/thinking_mode_probe.json): gemma honors
# enable_thinking:false (clean non-thinking output, all probe facts restated in ~150 tok);
# LFM2.5-8B-A1B cannot disable thinking (the flag leaks <think> in-band, truncating at
# this budget) and moved to the probe-teacher role instead.
MAX_TOKENS = 1024   # raised from 512 (2026-07-05, pre-gate calibration): full ACI notes hit
                    # the 512 cap mid-sentence (measured: out_len ~532 tok, tail truncated),
                    # killing ceiling-anchor validation on facts from later note sections.
                    # gemma finishes real notes in ~400-700 tok; 1024 is headroom, not a target.

UTILITY_EXECUTION_CONTRACT_VERSION = "ranker-v2-utility-execution-v1"
TASK_PROMPT_PIN_VERSION = "roundtrip-task-prompt-v1"

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
                            # Request scope: distinct rewrites use the served -np slots
                            # concurrently; identical requests still dedupe. The pin
                            # below stays single_flight=True (semantics unchanged).
                            single_flight=True, single_flight_scope="request")
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


def _task_prompt_pin(corpus: str) -> dict[str, str]:
    template = _template({"corpus": corpus})
    return {
        "version": TASK_PROMPT_PIN_VERSION,
        "corpus": corpus,
        "template_hash": stable_hash(template),
    }


def _remote_model_pin() -> dict[str, Any]:
    return {
        "model": RT_MODEL,
        "endpoint": RT_BASE_URL,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "thinking": False,
        "single_flight": True,
    }


def _ordered_action_vector(request: UtilityRequest) -> list[list[str]]:
    expected = [decision.decision_id for decision in request.document.policy_decisions]
    if set(request.action_vector) != set(expected):
        raise ValueError("utility request action vector must cover policy decisions exactly")
    return [
        [decision_id, str(request.action_vector[decision_id])]
        for decision_id in expected
    ]


def _cache_identity(
    request: UtilityRequest, doc_p: str, *, reader_refresh: bool,
) -> dict[str, Any]:
    artifact = request.utility_artifact
    if artifact.get("artifact_version") != "utility-assertions-v2":
        raise ValueError("utility request requires utility-assertions-v2")
    artifact_hash = artifact.get("artifact_hash")
    if not isinstance(artifact_hash, str) or not artifact_hash:
        raise ValueError("utility artifact lacks artifact_hash")
    expected_artifact_hash = stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    if artifact_hash != expected_artifact_hash:
        raise ValueError("utility artifact artifact_hash mismatch")
    if (
        not isinstance(request.environment_hash, str)
        or not request.environment_hash
        or artifact.get("environment_hash") != request.environment_hash
    ):
        raise ValueError("utility artifact environment_hash mismatch")
    return {
        "doc_id": request.document.doc_id,
        "ordered_action_vector": _ordered_action_vector(request),
        "doc_p_hash": stable_hash(doc_p),
        "environment_hash": request.environment_hash,
        "utility_artifact_hash": artifact_hash,
        "utility_binding": utility_binding(artifact, request.document.doc_id),
        "task_prompt_pin": _task_prompt_pin(request.document.corpus),
        "remote_model_pin": _remote_model_pin(),
        "extractor_pin": invert_implementation_pin(),
        "reader_pin": artifact.get("reader_pin"),
        "runtime_scorer_version": UTILITY_SCORER_VERSION,
        "execution_contract_version": UTILITY_EXECUTION_CONTRACT_VERSION,
        "reader_refresh": bool(reader_refresh),
    }


def _pmap_with_peak(
    function: Callable[[Any], Any], items: Sequence[Any], workers: int,
    *, metrics: dict[str, Any] | None = None, peak_name: str | None = None,
    transport: bool = False,
) -> list[Any]:
    if workers < 1:
        raise ValueError("worker count must be positive")
    if not items:
        return []
    from cloak.concurrent import pmap

    lock = threading.Lock()
    active = 0
    peak = 0

    def tracked(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if metrics is not None and peak_name is not None:
                metrics["peak_concurrency"][peak_name] = peak
            if metrics is not None and transport:
                metrics["transport_calls"] += 1
        try:
            return function(item)
        finally:
            with lock:
                active -= 1

    return pmap(tracked, list(items), workers=workers)


def _read_utility_work(item: Mapping, *, reader_refresh: bool) -> str:
    reader = read_context_set_batch if item["set_valued"] else read_context_batch
    kwargs = {"refresh": True} if reader_refresh else {}
    answers = reader(
        [item["question"]], item["context"], [item["reader_clause"]], **kwargs,
    )
    if len(answers) != 1:
        raise ValueError("reader returned the wrong number of answers")
    return str(answers[0])


def _validate_request_readers(request: UtilityRequest) -> None:
    artifact = request.utility_artifact
    validate_utility_reader_pin(read_context_batch, artifact)
    needs_set_reader = any(
        row.get("doc_id") == request.document.doc_id
        and row.get("status", "accepted") == "accepted"
        and row.get("family") == "context"
        and (row.get("answer_target") or {}).get("kind") == "linked_decision_set"
        for row in artifact.get("assertions", {}).values()
    )
    if needs_set_reader:
        validate_utility_reader_pin(read_context_set_batch, artifact)


def score_roundtrip_batch(
    requests: Sequence["UtilityRequest"],
    *,
    cache: UtilityCache,
    remote_workers: int,
    reader_workers: int,
    reader_refresh: bool = False,
) -> list[UtilityResult]:
    """Score complete utility vectors through staged, atomic miss execution."""
    metrics: dict[str, Any] = {
        "rollouts": len(requests),
        "context_assertions": 0,
        "reader_work_items": 0,
        "transport_calls": 0,
        "cache_hits": 0,
        "stage_latency_seconds": {
            name: 0.0 for name in (
                "render", "cache_lookup", "generation", "extraction",
                "deterministic_scoring", "reader", "finalize", "cache_append",
            )
        },
        "peak_concurrency": {"generation": 0, "reader": 0},
    }
    cache.last_batch_metrics = metrics
    if remote_workers < 1 or reader_workers < 1:
        raise ValueError("worker counts must be positive")
    if not requests:
        return []

    started = time.perf_counter()
    rendered = []
    try:
        for request in requests:
            _validate_request_readers(request)
            doc_p, replacements = assemble_action_vector(
                request.document, request.action_vector,
            )
            identity = _cache_identity(
                request, doc_p, reader_refresh=reader_refresh,
            )
            rendered.append((request, doc_p, replacements, identity))
    finally:
        metrics["stage_latency_seconds"]["render"] = time.perf_counter() - started

    unique: dict[str, tuple[UtilityRequest, str, list[dict], dict[str, Any]]] = {}
    request_keys = []
    for row in rendered:
        key = cache.request_identity(row[3])
        request_keys.append(key)
        unique.setdefault(key, row)

    started = time.perf_counter()
    resolved: dict[str, UtilityResult] = {}
    misses = []
    try:
        for key, row in unique.items():
            cached = cache.lookup(row[3])
            if cached is None:
                misses.append((key, row))
            else:
                resolved[key] = cached
                metrics["cache_hits"] += sum(value == key for value in request_keys)
    finally:
        metrics["stage_latency_seconds"]["cache_lookup"] = time.perf_counter() - started
    if not misses:
        return [resolved[key] for key in request_keys]

    remote = _remote()
    started = time.perf_counter()
    try:
        generated = _pmap_with_peak(
            lambda item: remote.generate(
                _template({"corpus": item[1][0].document.corpus}).format(doc=item[1][1])
            ),
            misses,
            remote_workers,
            metrics=metrics,
            peak_name="generation",
            transport=True,
        )
    finally:
        metrics["stage_latency_seconds"]["generation"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        extracted = _pmap_with_peak(
            lambda indexed: invert(
                generated[indexed[0]], misses[indexed[0]][1][2],
            )[0],
            list(enumerate(misses)),
            remote_workers,
        )
    finally:
        metrics["stage_latency_seconds"]["extraction"] = time.perf_counter() - started

    started = time.perf_counter()
    prepared = []
    try:
        for index, (_key, row) in enumerate(misses):
            request, doc_p, _replacements, _identity = row
            scoring = prepare_utility_scoring(
                request.utility_artifact,
                request.document.doc_id,
                doc_p=doc_p,
                out_final=extracted[index],
            )
            prepared.append(scoring)
    finally:
        metrics["stage_latency_seconds"]["deterministic_scoring"] = (
            time.perf_counter() - started
        )

    work_queue = [
        (rollout_index, item)
        for rollout_index, scoring in enumerate(prepared)
        for item in scoring["context_work"]
    ]
    metrics["context_assertions"] = len(work_queue)
    metrics["reader_work_items"] = len(work_queue)
    started = time.perf_counter()
    try:
        answers = _pmap_with_peak(
            lambda indexed: (
                indexed[0],
                _read_utility_work(indexed[1], reader_refresh=reader_refresh),
            ),
            work_queue,
            reader_workers,
            metrics=metrics,
            peak_name="reader",
            transport=True,
        )
    finally:
        metrics["stage_latency_seconds"]["reader"] = time.perf_counter() - started

    answers_by_rollout: list[list[str]] = [[] for _ in misses]
    for rollout_index, answer in answers:
        answers_by_rollout[rollout_index].append(answer)
    started = time.perf_counter()
    staged = []
    try:
        for index, (key, row) in enumerate(misses):
            request, doc_p, _replacements, identity = row
            scored = finalize_utility_scoring(
                prepared[index], answers_by_rollout[index],
            )
            result = make_result(
                doc_id=request.document.doc_id,
                action_vector={
                    decision_id: action_id
                    for decision_id, action_id in identity["ordered_action_vector"]
                },
                doc_p=doc_p,
                out_p=generated[index],
                out_final=extracted[index],
                component_scores=scored["component_scores"],
                utility=scored["utility"],
            )
            staged.append((identity, result))
            resolved[key] = result
    finally:
        metrics["stage_latency_seconds"]["finalize"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        cache.store_many(staged)
    finally:
        metrics["stage_latency_seconds"]["cache_append"] = time.perf_counter() - started
    return [resolved[key] for key in request_keys]


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
            score_kwargs = {
                "doc_p": j["doc_p"],
                "out_final": out_final,
                "reader": read_context_batch,
                "set_reader": read_context_set_batch,
            }
            if reader_refresh:
                score_kwargs["reader_refresh"] = True
            scored = score_utility(
                j["utility_artifact"], j["doc_id"], **score_kwargs,
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
