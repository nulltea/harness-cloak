"""Generative-reader sweep with scorer v2. Readers: local (transformers) + served (proxy).

Readers this round (all non-thinking, greedy/temp0, grounded+abstaining prompt, batched):
  - Qwen/Qwen2.5-1.5B-Instruct  (local)
  - gemma 4 (E4B)               (SERVED on :8060, batched via pmap — reward model as reader,
                                 the workflow-alignment test done with the instruct model)
  - Qwen/Qwen3.5-0.8B           (local)

Scorer v2 (score fn): canon-normalize -> NUMBER GATE (gold numbers must appear in the answer,
else 0; kills "10 mg"~="40 milligrams") -> containment (gold tokens subset of answer -> 1.0)
-> acronym match (CHF == initials of "Congestive Heart Failure") -> token-F1 fallback.
Residual (under-scores): non-initial abbrevs (HTN), pure synonyms (renal==kidney).

Metrics: ceil_pass (recall), floor_pass (hallucination/leak), reader_miss on present, abstain.
Gemma anchors cached; local models batch per doc; the served reader batches via pmap workers.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
       scripts/spikes/generative_reader_sweep.py [--per-corpus 5]
"""
import argparse
import gc
import json
import re
from pathlib import Path

import torch
from train_ranker import assemble

from cloak.corpora import load_task_docs
from cloak.train.reward import canon
from cloak.train.roundtrip import RT_BASE_URL, roundtrip_batch

ENV = Path("data/ranker_env_full.json")
ARMS = Path("data/task_arms_full.json")
CORPORA = ("clinical", "lexsum", "wikibio")
TH = 0.5
READERS = [("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "local"),
           ("gemma-4-E4B", "gemma 4 (E4B)", "served"),
           ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "local")]
PROMPT = ("Answer the question using ONLY the note below. Reply with the shortest exact "
          "answer copied from the note (a name, value, number, or phrase). If the note does "
          "not contain the answer, reply exactly: NONE.\n\nNote:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def score(pred, gold):
    """Scorer v2 — see module docstring."""
    p = re.findall(r"\w+", canon(pred)); g = re.findall(r"\w+", canon(gold))
    if not g:
        return float(not p)
    if not p:
        return 0.0
    gnum = [t for t in g if t.isdigit()]
    pnum = {t for t in p if t.isdigit()}
    if gnum and not all(n in pnum for n in gnum):   # number gate: wrong number -> no credit
        return 0.0
    ps, gs = set(p), set(g)
    if gs <= ps:                                     # containment
        return 1.0
    def acro(short, long_):                          # CHF == c+h+f of the words
        return len(short) == 1 and len(short[0]) >= 2 and short[0] == "".join(w[0] for w in long_)
    if acro(p, g) or acro(g, p):
        return 1.0
    common = sum(min(p.count(t), g.count(t)) for t in gs)
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def parse(raw):
    a = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    return "" if a.upper().strip(".:` ") == "NONE" else a


def load_local(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    return tok, m.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def ask_local(tok, model, questions, ctx):
    prompts = []
    for q in questions:
        msg = [{"role": "user", "content": PROMPT.format(ctx=ctx, q=q)}]
        try:
            prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True,
                                                   enable_thinking=False, tokenize=False))
        except TypeError:
            prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
              add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return [parse(r) for r in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                               skip_special_tokens=True)]


def make_served(model_id):
    from inferdpt.llm import LLMClient
    return LLMClient(model_id, base_url=RT_BASE_URL, api_key="x", temperature=0.0,
                     max_tokens=32, extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def ask_served(client, questions, ctx):
    from inferdpt.pipeline import pmap
    return [parse(r) for r in pmap(lambda q: client.generate(PROMPT.format(ctx=ctx, q=q)),
                                   questions, workers=6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=5)
    args = ap.parse_args()
    env = json.loads(ENV.read_text())["corpora"]
    arms = json.loads(ARMS.read_text())

    docs = []
    for corpus in CORPORA:
        texts = {x["id"]: x["text"] for x in load_task_docs(corpus)}
        picked = [(i, d) for i, d in env[corpus].items()
                  if d.get("spans") and d.get("probes", {}).get("train")][:args.per_corpus]
        for doc_id, d in picked:
            spans = d["spans"]
            ph = {s["surface"].lower(): next(a for a in s["actions"]
                                             if a["mode"] == "placeholder") for s in spans}
            lo_doc, lo_R = assemble(texts[doc_id], arms[corpus][doc_id]["tau_walk"][1], spans, ph)
            hi_r, lo_r = roundtrip_batch(
                [{"corpus": corpus, "doc_p": texts[doc_id], "R": [], "probes": []},
                 {"corpus": corpus, "doc_p": lo_doc, "R": lo_R, "probes": []}], workers=2)
            docs.append((corpus, doc_id, hi_r["out_final"], lo_r["out_final"], d["probes"]["train"]))

    rows = []
    for name, mid, kind in READERS:
        print(f"... {name}", flush=True)
        if kind == "local":
            tok, model = load_local(mid)
            ask = lambda qs, c: ask_local(tok, model, qs, c)
        else:
            client = make_served(mid)
            ask = lambda qs, c: ask_served(client, qs, c)
        a = {"n": 0, "hi_pass": 0, "lo_fail": 0, "present": 0, "miss": 0, "abstain": 0}
        for corpus, doc_id, out_hi, out_lo, probes in docs:
            qs = [p["question"] for p in probes]
            A_hi, A_lo = ask(qs, out_hi), ask(qs, out_lo)
            for p, ans_hi, ans_lo in zip(probes, A_hi, A_lo):
                g = p["surface"]
                present = canon(g) in canon(out_hi)
                hi, lo = score(ans_hi, g), score(ans_lo, g)
                a["n"] += 1; a["hi_pass"] += hi >= TH; a["lo_fail"] += lo < TH
                a["present"] += present; a["abstain"] += ans_hi == ""
                if present and hi < TH:
                    a["miss"] += 1
        rows.append((name, a))
        if kind == "local":
            del model, tok; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'reader':<15}{'ceil_pass':<11}{'floor_pass':<12}{'reader_miss':<13}{'abstain':<10}")
    for name, a in rows:
        n = max(a["n"], 1); pres = max(a["present"], 1)
        print(f"{name:<15}{a['hi_pass']/n:<11.1%}{a['lo_fail']/n:<12.1%}"
              f"{a['miss']/pres:<13.1%}{a['abstain']/n:<10.1%}")


if __name__ == "__main__":
    main()
