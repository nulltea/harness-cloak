"""Probe-question generation for the surrogate reward (Phase 0, once per document).

For each restated span (R surface found in the gold output) the local teacher writes one
natural question whose answer is that span. Questions are cached, so the per-candidate
reward loop stays model-free (only the extractive reader runs at reward time). Natural
questions are required: cloze phrasings are OOD for SQuAD2 readers and abstain (measured
2026-07-02, reward.py self-check history).

Teacher = LFM2.5-8B-A1B on the local llama-swap (user re-pin 2026-07-05: gemma 4 (E4B)
became the round-trip reward model, and the teacher must be a different family —
teacher != reward model). Legacy gemma-authored questions live in the pre-re-pin cache;
set that file aside (do not mix teachers in one cache) before rebuilding.
Cache: data/surrogate_probes.json, keyed by doc id.
"""
import json
import re
from pathlib import Path

from cloak.train.reward import restated_probes

CACHE = Path("data/surrogate_probes.json")
# Escalated LFM2.5 -> Qwen3.6 (2026-07-05, the spec's pre-registered trigger): probe-health
# measured LFM question quality as the binder — ambiguous multi-answer questions ("What
# condition is listed...?" over a 5-item list), mistargeted answers, reader abstention;
# ceiling pass ~23-30%, every doc under the 3-probe floor. Teacher-tagged cache retires the
# LFM entries automatically. Family separation holds (teacher != gemma reward model); the
# second-remote eval arm moves to LFM2.5 (spec components table).
TEACHER_MODEL = "Qwen3.6-35B-A3B"

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
    n_legacy = 0
    todo = []
    for d in docs:
        # only current-teacher entries are reusable; legacy / other-teacher entries (incl.
        # ones with no "teacher" field) are ignored and their surfaces go back into todo, so
        # teachers can never mix in one cache without a manual mv
        have = set()
        for p in cache.get(d["id"], []):
            if p.get("teacher") == TEACHER_MODEL:
                have.add(p["surface"])
            else:
                n_legacy += 1
        for p in restated_probes(R_of[d["id"]], refs_of(d)[0]):
            if p["entry"]["surface"] not in have:
                todo.append({"doc_id": d["id"], "surface": p["entry"]["surface"],
                             "sent": p["gold_sent"]})
    if n_legacy:
        print(f"probes: ignored {n_legacy} legacy/other-teacher cache entries "
              f"(teacher != {TEACHER_MODEL})", flush=True)
    if todo:
        # Qwen3.6 honors enable_thinking:False (unlike LFM2.5, whose unconditional thinking
        # was one reason its questions underperformed) — pass it so content is the question.
        teacher = LLMClient(TEACHER_MODEL, base_url="http://localhost:8060/v1", api_key="x",
                            temperature=0.0, max_tokens=256,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        replies = pmap(lambda t: teacher.generate(
            PROMPT.format(answer=t["surface"], sent=t["sent"])), todo, workers=workers)
        n_lost = 0
        for t, r in zip(todo, replies):
            reply = (r or "").strip()
            # <think> leak (thinking teacher) or empty reply -> unusable, before extraction
            if not reply or "<think>" in reply:
                n_lost += 1
                continue
            q = re.sub(r"^[\"']|[\"']$", "", reply.splitlines()[0].strip())
            if _valid(q, t["surface"]):
                cache.setdefault(t["doc_id"], []).append(
                    {"surface": t["surface"], "question": q, "teacher": TEACHER_MODEL})
            else:
                n_lost += 1
        print(f"probes: {n_lost}/{len(todo)} teacher replies invalid/empty (no probe kept)",
              flush=True)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2))
    return {d["id"]: [p for p in cache.get(d["id"], []) if p.get("teacher") == TEACHER_MODEL]
            for d in docs}


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
