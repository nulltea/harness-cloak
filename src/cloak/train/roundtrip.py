"""Round-trip reward (spec docs/specs/RL/roundtrip-ranker-infiller.md, Phase 1).

R_rt = realized fact recall (graded mean token-F1) on out_final over a doc's train-split
probes, where out_final = invert(Remote(task_prompt(doc_p)), R). Deterministic given doc_p:
pinned model, temperature 0, content-addressed disk cache (INFERDPT_LLM_CACHE) — the
determinism is load-bearing (cache = reward memoization = ExIt pool; spec "one subtlety").
"""
import os

from cloak.extract import invert
from cloak.tasks import TASK_TEMPLATE
from cloak.train.reward import fact_f1s

RT_MODEL = "LFM2.5-8B-A1B"   # THE pin (spec components table); changing it re-gates
MAX_TOKENS = 512

_client = None


def _remote():
    global _client
    if _client is None:
        from inferdpt.llm import LLMClient
        assert os.getenv("INFERDPT_LLM_CACHE"), \
            "round-trip reward requires INFERDPT_LLM_CACHE (determinism + cost)"
        _client = LLMClient(RT_MODEL, temperature=0.0, max_tokens=MAX_TOKENS,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return _client


def roundtrip_batch(jobs: list[dict], workers: int = 8) -> list[dict]:
    """jobs: [{corpus, doc_p, R, probes}] -> [{out_p, out_final, f1s, recall}].
    recall = graded mean token-F1 (the deployed fact_recall), None when a job has no probes."""
    from inferdpt.pipeline import pmap
    remote = _remote()
    outs = pmap(lambda j: remote.generate(
        TASK_TEMPLATE[j["corpus"]].format(doc=j["doc_p"])), jobs, workers=workers)
    res = []
    for j, op in zip(jobs, outs):
        out_final, _ = invert(op, j["R"])
        f1s = fact_f1s(out_final, j["probes"])
        res.append({"out_p": op, "out_final": out_final, "f1s": f1s,
                    "recall": (sum(f1s) / len(f1s)) if f1s else None})
    return res


if __name__ == "__main__":
    # LIVE smoke (hits the proxy once; requires INFERDPT_LLM_CACHE and the proxy up):
    #   INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src .venv/bin/python -m cloak.train.roundtrip
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
