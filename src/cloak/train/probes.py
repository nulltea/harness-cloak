"""Probe-question generation for the surrogate reward (Phase 0, once per document).

For each restated span (R surface found in the gold output) the local teacher writes one
natural question whose answer is that span. Questions are cached, so the per-candidate
reward loop stays model-free (only the extractive reader runs at reward time). Natural
questions are required: cloze phrasings are OOD for SQuAD2 readers and abstain (measured
2026-07-02, reward.py self-check history).

Teacher = gemma 4 (E4B) on the local llama-swap, same pattern as lattice.teacher_lattices.
Cache: data/surrogate_probes.json, keyed by doc id.
"""
import json
import re
from pathlib import Path

from cloak.train.reward import restated_probes

CACHE = Path("data/surrogate_probes.json")

PROMPT = """Write one short factual question about the text below whose exact answer is "{answer}".
The question must be answerable from the text alone and must not contain "{answer}" itself.

Text: {sent}

Reply with the question only."""


def _valid(q: str, answer: str) -> bool:
    q = q.strip()
    return bool(q) and q.endswith("?") and len(q) < 200 and answer.lower() not in q.lower()


def probes_for_docs(docs: list[dict], R_of: dict[str, list[dict]], workers: int = 6) -> dict:
    """docs: corpora.load_task_docs rows; R_of: doc id -> R. Returns {doc_id: [probe]},
    probe = {"surface", "question"}. Teacher-fills the cache for missing (doc, surface)."""
    from cloak.corpora import refs_of
    from inferdpt.llm import LLMClient
    from inferdpt.pipeline import pmap

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = []
    for d in docs:
        have = {p["surface"] for p in cache.get(d["id"], [])}
        for p in restated_probes(R_of[d["id"]], refs_of(d)[0]):
            if p["entry"]["surface"] not in have:
                todo.append({"doc_id": d["id"], "surface": p["entry"]["surface"],
                             "sent": p["gold_sent"]})
    if todo:
        e4b = LLMClient("gemma 4 (E4B)", base_url="http://localhost:8060/v1", api_key="x",
                        temperature=0.0, max_tokens=100,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        replies = pmap(lambda t: e4b.generate(
            PROMPT.format(answer=t["surface"], sent=t["sent"])), todo, workers=workers)
        for t, r in zip(todo, replies):
            q = re.sub(r"^[\"']|[\"']$", "", (r or "").strip().splitlines()[0].strip()) \
                if (r or "").strip() else ""
            if _valid(q, t["surface"]):
                cache.setdefault(t["doc_id"], []).append(
                    {"surface": t["surface"], "question": q})
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2))
    return {d["id"]: cache.get(d["id"], []) for d in docs}


if __name__ == "__main__":
    # wiring check without the teacher: cache round trip + restated-probe extraction
    R = [{"action": "generalize", "surface": "Oslo", "replacement": "a Norwegian city"}]
    doc = {"id": "_selfcheck", "gold_ref": "The nurse from Oslo presented with chest pain."}
    from cloak.corpora import refs_of
    ps = restated_probes(R, refs_of(doc)[0])
    assert len(ps) == 1 and ps[0]["entry"]["surface"] == "Oslo", ps
    assert _valid("Where is the nurse from?", "Oslo")
    assert not _valid("Is Oslo nice?", "Oslo") and not _valid("no question mark", "x")
    print("probes.py self-check OK (teacher path exercised by scripts/surrogate_validation.py)")
