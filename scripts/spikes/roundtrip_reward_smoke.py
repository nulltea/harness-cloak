"""3-doc full round-trip + reward trace — component check incl. the QA reader.

One trainable doc per corpus (clinical, lexsum, wikibio) taken through the ENTIRE reward
path with the behavior-clone (floor-walk) action set:

    assemble(doc, R_walk, spans, BC-choice)   # env mask -> doc_p, R
      -> Remote(task_prompt[corpus](doc_p))    # gemma 4 (E4B), the reward's LLM
      -> invert(out_p, R)                      # deployed extractor
      -> _qa_answer(question, out_final)        # deepset/roberta-base-squad2 QA reader
      -> token_f1(answer, surface)              # per-probe, then per-fact max mean = R_rt

Prints the full trace per doc so every component's output is eyeballable: doc_p, R map,
out_p, out_final, and per-probe (question / gold surface / reader answer / f1), then the
reward. Hits the proxy once per doc (BC doc_p != anchors, so cache-cold) — needs the proxy
free and INFERDPT_LLM_CACHE set.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/roundtrip_reward_smoke.py
"""
import json
from pathlib import Path

from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.train.reward import _max_by_fact, _qa_answer, token_f1
from cloak.train.roundtrip import roundtrip_batch

ENV = Path("data/ranker_env_full.json")
ARMS = Path("data/task_arms_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")


def pick_trainable(env_corpus: dict) -> tuple[str, dict]:
    """First doc with train probes AND spans."""
    for doc_id, d in env_corpus.items():
        if d.get("spans") and d.get("probes", {}).get("train"):
            return doc_id, d
    raise SystemExit("no trainable doc")


def main():
    env = json.loads(ENV.read_text())["corpora"]
    arms = json.loads(ARMS.read_text())

    for corpus in CORPORA:
        doc_id, d = pick_trainable(env[corpus])
        text = next(x["text"] for x in load_task_docs(corpus) if x["id"] == doc_id)
        R_walk = arms[corpus][doc_id]["tau_walk"][1]
        # behavior-clone / floor-walk action set: each span's stored bc_action
        choice = {s["surface"].lower(): s["actions"][s["bc_action"]] for s in d["spans"]}
        doc_p, R = assemble(text, R_walk, d["spans"], choice)
        probes = d["probes"]["train"]

        res = roundtrip_batch([{"corpus": corpus, "doc_p": doc_p, "R": R,
                                "probes": probes}], workers=1)[0]

        print("\n" + "=" * 88)
        print(f"[{corpus}] {doc_id}  spans={len(d['spans'])}  train_probes={len(probes)}")
        print("-" * 88)
        print(f"doc_orig[:300]: {text[:300]!r}")
        print(f"doc_p[:300]:    {doc_p[:300]!r}")
        print("R (surface -> replacement):")
        for e in R:
            print(f"    {e['surface']!r} -> {e['replacement']!r}  ({e.get('action')})")
        print(f"out_p[:400]:     {res['out_p'][:400]!r}")
        print(f"out_final[:400]: {res['out_final'][:400]!r}")
        print("QA reader per probe (question | gold surface | reader answer | f1):")
        for p, f1 in zip(probes, res["f1s"]):
            ans = _qa_answer(p["question"], res["out_final"])
            assert abs(token_f1(ans, p["surface"]) - f1) < 1e-6, "reader mismatch vs fact_f1s"
            print(f"    q={p['question'][:60]!r}")
            print(f"      gold={p['surface']!r}  reader={ans!r}  f1={f1:.3f}")
        by_fact = _max_by_fact(probes, res["f1s"])
        print(f"per-fact max: {[round(v, 3) for v in by_fact.values()]}")
        print(f"==> R_rt (reward, per-fact-max mean) = {res['recall']}")


if __name__ == "__main__":
    main()
