"""Classify Qwen3.5-0.8B's floor leaks: hallucination vs echo vs survivor.

For each probe, read the answer from out_lo (the all-placeholder floor, after invert). A
"leak" = score_new(ans, gold) >= TH. Classify:
  - hallucination : gold NOT in out_lo and answer NOT in out_lo -> reader invented it
                    (the generative-only risk; extractive can't do this)
  - echo          : gold WAS a redacted span AND appears in out_lo -> placeholder
                    round-tripped back through invert() (pipeline floor-leak, not reader)
  - survivor      : gold in out_lo but was never a redacted span -> detection miss

hallucination-heavy => the reader is the problem (disqualify / needs abstain-tuning).
echo/survivor-heavy => the floor itself leaks; probe-validation floor-reject already drops
those, so it's not a reader fault.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/qwen_floor_leak_audit.py [--per-corpus 10] [--examples 8]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from generative_reader_sweep import ask_gen, load, score_new
from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.train.reward import canon
from cloak.train.roundtrip import roundtrip_batch

ENV = Path("data/ranker_env_full.json")
ARMS = Path("data/task_arms_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")
TH = 0.5


def window(text, needle, w=70):
    i = text.lower().find(needle.lower())
    return text[max(0, i - w):i + w].replace("\n", " ") if i >= 0 else "(not in out_lo)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=10)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    env = json.loads(ENV.read_text())["corpora"]
    arms = json.loads(ARMS.read_text())
    tok, model = load("Qwen/Qwen3.5-0.8B", "gen")

    tally = defaultdict(int)
    ex = defaultdict(list)
    n_leak = 0
    for corpus in CORPORA:
        texts = {x["id"]: x["text"] for x in load_task_docs(corpus)}
        picked = [(i, d) for i, d in env[corpus].items()
                  if d.get("spans") and d.get("probes", {}).get("train")][:args.per_corpus]
        for doc_id, d in picked:
            spans = d["spans"]
            redacted = {canon(s["surface"]) for s in spans}
            ph = {s["surface"].lower(): next(a for a in s["actions"]
                                             if a["mode"] == "placeholder") for s in spans}
            lo_doc, lo_R = assemble(texts[doc_id], arms[corpus][doc_id]["tau_walk"][1], spans, ph)
            out_lo = roundtrip_batch(
                [{"corpus": corpus, "doc_p": lo_doc, "R": lo_R, "probes": []}], workers=1
            )[0]["out_final"]
            qs = [p["question"] for p in d["probes"]["train"]]
            ans = ask_gen(tok, model, qs, out_lo)
            for p, a in zip(d["probes"]["train"], ans):
                g = p["surface"]
                if score_new(a, g) < TH:
                    continue
                n_leak += 1
                gold_in = canon(g) in canon(out_lo)
                ans_in = bool(a) and canon(a) in canon(out_lo)
                if not gold_in and not ans_in:
                    cat = "hallucination"
                elif canon(g) in redacted and gold_in:
                    cat = "echo"
                else:
                    cat = "survivor"
                tally[cat] += 1
                ex[cat].append((corpus, g, p["question"], a, window(out_lo, a or g)))

    print(f"\nFloor leaks (lo_f1 >= {TH}): {n_leak}")
    for c in ("hallucination", "echo", "survivor"):
        print(f"  {c:<14}{tally[c]:>4}  ({tally[c]/max(n_leak,1):.0%})")
    for c in ("hallucination", "echo", "survivor"):
        print("\n" + "=" * 92 + f"\n### {c}  ({tally[c]})")
        for corpus, g, q, a, win in ex[c][:args.examples]:
            print(f"[{corpus}] gold={g!r}  reader_on_floor={a!r}")
            print(f"    q:      {q[:75]!r}")
            print(f"    out_lo: {win!r}")


if __name__ == "__main__":
    main()
