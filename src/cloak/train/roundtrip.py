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
from cloak.tasks import TASK_TEMPLATE
from cloak.train.reward import _max_by_fact, fact_f1s

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


def roundtrip_batch(
    jobs: list[dict],
    workers: int = 6,
    extractor_models: dict | None = None,
    reader_refresh: bool = False,
) -> list[dict]:
    """jobs: [{corpus, doc_p, R, probes}] -> [{out_p, out_final, f1s, recall}].
    recall = deployed fact_recall (per-fact max, mean over facts), None when no probes.
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
        op = remote.generate(TASK_TEMPLATE[j["corpus"]].format(doc=j["doc_p"]))
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
