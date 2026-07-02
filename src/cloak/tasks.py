"""Task construction over SynthPAI docs: summarization + teacher-generated QA.

QA pairs are generated once from doc_orig by the local teacher (llama-swap Qwen3.6),
cached to data/latticecloak_qa.json — gold answers are grounded in doc_orig, so QA scoring
needs no model. Plan: docs/plans/2026-07-02-d1-prototype-implementation.md.
"""
import json
import re
from pathlib import Path

QA_PATH = Path("data/latticecloak_qa.json")

SUMMARIZE = ("Summarize this Reddit user's interests, life situation and personality "
             "in 3-4 sentences, based only on their comments below.\n\nComments:\n{doc}")
ANSWER = ("Answer the question briefly (one sentence), based only on this Reddit user's "
          "comments below.\n\nComments:\n{doc}\n\nQuestion: {question}")
QA_GEN = ("Read this Reddit user's comments. Write exactly 3 factual questions about the "
          "user's life, habits or surroundings that are answerable from the text, each with "
          "a short answer quoted or paraphrased from the text. Reply with ONLY a JSON list: "
          '[{{"q": "...", "a": "..."}}, ...]\n\nComments:\n{doc}')

# Task-oriented templates whose output must restate the input's entities (docs/specs/benchmarks.md).
CLINICAL_NOTE = ("Write a clinical visit note for the following doctor-patient dialogue, using "
                 "standard note sections. Base it only on the dialogue.\n\nDialogue:\n{doc}")
EMAIL_SUBJECT = ("Write a concise subject line (a few words) for the following email. "
                 "Reply with only the subject line.\n\nEmail:\n{doc}")
EMAIL_REPLY = ("Write a brief reply to the following email. Reply with only the message "
               "body.\n\nEmail:\n{doc}")
TASK_TEMPLATE = {"aci": CLINICAL_NOTE, "mts": CLINICAL_NOTE, "clinical": CLINICAL_NOTE,
                 "aeslc": EMAIL_SUBJECT, "enron": EMAIL_REPLY}


def _teacher():
    from inferdpt.llm import LLMClient
    return LLMClient("Qwen3.6-35B-A3B", base_url="http://localhost:8060/v1", api_key="x",
                     temperature=0.0, max_tokens=400,
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def qa_pairs(docs: list[dict], workers: int = 6) -> dict[str, list[dict]]:
    """{author: [{q, a}, ...]}; generated once, cached."""
    from concurrent.futures import ThreadPoolExecutor
    cache = json.loads(QA_PATH.read_text()) if QA_PATH.exists() else {}
    todo = [d for d in docs if d["author"] not in cache]
    if todo:
        teacher = _teacher()
        with ThreadPoolExecutor(workers) as ex:
            replies = list(ex.map(lambda d: teacher.generate(QA_GEN.format(doc=d["text"])), todo))
        for d, r in zip(todo, replies):
            m = re.search(r"\[.*\]", r, re.DOTALL)
            try:
                pairs = json.loads(m.group()) if m else []
            except json.JSONDecodeError:
                pairs = []
            cache[d["author"]] = [p for p in pairs if isinstance(p, dict) and p.get("q") and p.get("a")][:3]
        QA_PATH.parent.mkdir(exist_ok=True)
        QA_PATH.write_text(json.dumps(cache, indent=2))
    return cache


def doc_tasks(doc: dict, qa: dict) -> list[dict]:
    """[{task_id, prompt_template, question?, gold?}] — prompt filled with doc/doc_p later."""
    tasks = [{"task_id": "summarize", "template": SUMMARIZE}]
    for i, p in enumerate(qa.get(doc["author"], [])):
        tasks.append({"task_id": f"qa{i}", "template": ANSWER, "question": p["q"], "gold": p["a"]})
    return tasks


def fill(task: dict, text: str) -> str:
    return task["template"].format(doc=text, question=task.get("question", ""))


if __name__ == "__main__":
    from cloak.synthpai import load_docs
    docs = load_docs(2)
    qa = qa_pairs(docs)
    for d in docs:
        ts = doc_tasks(d, qa)
        assert ts[0]["task_id"] == "summarize" and len(ts) >= 1
        print(d["author"], [(t["task_id"], t.get("question", "")[:50]) for t in ts])
    print("tasks.py self-check OK")
