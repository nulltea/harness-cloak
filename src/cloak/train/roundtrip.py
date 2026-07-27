"""Round-trip reward (spec docs/specs/RL/roundtrip-ranker-infiller.md, Phase 1).

R_rt = realized fact recall (graded mean token-F1) on out_final over a doc's train-split
probes, where out_final = invert(Remote(task_prompt(doc_p)), R). Deterministic given doc_p:
pinned model, temperature 0, single-flight generation, and content-addressed disk cache
(CLOAK_LLM_CACHE) — the determinism is load-bearing (cache = reward memoization = ExIt
pool; spec §Determinism under concurrency).

THE reward pin (changing any re-gates): RT_MODEL = "medgemma-4b-it" served at
RT_BASE_URL = "http://localhost:8060/v1", temperature 0, max_tokens 1024, non-thinking.
The extractor is part of the reward pin (`invert`, hashed into `extractor_pin`): cached
rewards are valid only under the pin they were produced with.
"""
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from cloak.extract import invert, invert_implementation_pin
from cloak.tasks import SCHEMA_TEMPLATE, TASK_TEMPLATE
from cloak.train.interactive_ranker import assemble_action_vector
from cloak.train.qa_scoring import (
    UTILITY_SCORER_VERSION,
    finalize_utility_scoring,
    prepare_utility_scoring,
    read_context_batch,
    read_context_set_batch,
    validate_utility_reader_pin,
)
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


def _template(job: dict) -> str:
    wants_schema = job.get("template") == "schema" or job.get("schema")
    if wants_schema and job["corpus"] in SCHEMA_TEMPLATE:
        return SCHEMA_TEMPLATE[job["corpus"]]
    return TASK_TEMPLATE[job["corpus"]]


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
