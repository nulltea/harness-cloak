"""Spike: does keeping attribute content (conditions/meds/doses) verbatim help an LLM
attacker recover the IDENTITY spans?

Per (doc, arm in {tau_walk, identity_only, all_placeholder}): one prompt gives the full
doc_p and asks for 5 guesses per identity slot (direct identifiers + LOC/ORG/DATETIME
quasi spans — the same slot set across arms, so the only variable is what the attacker
can see around them). Score fuzzy hit@5 vs the original surface. If
hit(identity_only) ≈ hit(all_placeholder), keeping content does not leak identity at
this referee's strength; if it jumps, the kept content is a re-identification channel.

Referee = local Qwen3.6 (free proxy) — the shootout's weaker referee; treat absolute
rates as a lower bound on attack success, compare arms relatively.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/identity_attack.py
"""
import json
import re
import time
from pathlib import Path

from rapidfuzz import fuzz

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap
from reward_gate import IDENTITY_TYPES, identity_only_choice
from train_ranker import assemble

ATTACK_MODEL = "Qwen3.6-35B-A3B"
ARMS = ["tau_walk", "identity_only", "all_placeholder"]

PROMPT = """You are auditing an anonymization system. Below is an anonymized document. \
Some phrases were replaced by placeholder tokens (like <PERSON_1>) or by more generic \
phrases. For each numbered item, guess what the ORIGINAL text most likely was. Use the \
document context and your world knowledge. Give exactly 5 guesses per item, semicolon-\
separated, most likely first. Answer with one line per item, format: `1) g1; g2; g3; g4; g5`

Document:
{doc_p}

Items (as they appear in the document, with the original's type):
{items}"""


def main():
    t0 = time.time()
    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    llm = LLMClient(ATTACK_MODEL, temperature=0.0, max_tokens=600,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    jobs = []
    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
        for doc_id, d in per_doc.items():
            if not d["trainable"] or not d["spans"]:
                continue
            a = art[corpus][doc_id]
            R_walk = a["tau_walk"][1]
            # identity slots: direct identifiers (no lattice) + identity-type quasi spans
            targets, seen = [], set()
            for e in R_walk:
                if not e.get("lattice") and e["surface"].lower() not in seen:
                    seen.add(e["surface"].lower())
                    targets.append({"surface": e["surface"], "type": e["type"]})
            for s in d["spans"]:
                if s["type"] in IDENTITY_TYPES and s["surface"].lower() not in seen:
                    seen.add(s["surface"].lower())
                    targets.append({"surface": s["surface"], "type": s["type"]})
            if not targets:
                continue
            ph_choice = {s["surface"].lower(): s["actions"][-1] for s in d["spans"]}
            arm_docs = {
                "tau_walk": tuple(a["tau_walk"]),
                "identity_only": assemble(texts[doc_id], R_walk, d["spans"],
                                          identity_only_choice(d["spans"]))[:2],
                "all_placeholder": assemble(texts[doc_id], R_walk, d["spans"],
                                            ph_choice)[:2],
            }
            for arm, (doc_p, R) in arm_docs.items():
                rep = {e["surface"].lower(): e["replacement"] for e in R}
                items = [{"shown": rep.get(t["surface"].lower(), "<removed>"),
                          "surface": t["surface"], "type": t["type"]} for t in targets]
                jobs.append({"doc": doc_id, "corpus": corpus, "arm": arm,
                             "doc_p": doc_p, "items": items})

    def ask(j):
        listing = "\n".join(f'{i + 1}) "{it["shown"]}" (originally a {it["type"]})'
                            for i, it in enumerate(j["items"]))
        return llm.generate(PROMPT.format(doc_p=j["doc_p"], items=listing))

    replies = pmap(ask, jobs, workers=8)

    rows = []
    for j, rep in zip(jobs, replies):
        lines = {}
        for m in re.finditer(r"^\s*(\d+)\)\s*(.+)$", rep or "", re.MULTILINE):
            lines[int(m.group(1))] = [g.strip() for g in m.group(2).split(";")][:5]
        for i, it in enumerate(j["items"]):
            guesses = lines.get(i + 1, [])
            hit = any(fuzz.partial_ratio(g.lower(), it["surface"].lower()) >= 85
                      for g in guesses if g)
            rows.append({"doc": j["doc"], "corpus": j["corpus"], "arm": j["arm"],
                         "surface": it["surface"], "type": it["type"],
                         "direct": it["type"] not in IDENTITY_TYPES,
                         "hit5": hit, "guesses": guesses})

    report = {"attacker": ATTACK_MODEL, "arms": {}}
    for arm in ARMS:
        ar = [r for r in rows if r["arm"] == arm]
        by = lambda pred: (lambda xs: round(sum(r["hit5"] for r in xs) / len(xs), 3)
                           if xs else None)([r for r in ar if pred(r)])
        report["arms"][arm] = {"n": len(ar), "hit5": by(lambda r: True),
                               "hit5_direct_ids": by(lambda r: r["direct"]),
                               "hit5_quasi_identity": by(lambda r: not r["direct"])}
        print(f"{arm:16s} {report['arms'][arm]}", flush=True)
    report["rows"] = rows
    out = Path("results/identity_attack.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wall {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
