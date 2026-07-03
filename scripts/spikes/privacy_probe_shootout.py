"""Spike: which local candidate-sensitive privacy probe tracks a real LLM attacker?

Items = (span, lattice level) pairs from the arms artifact (surface, type, context sentence,
fill). Ground truth = remote frontier attacker (gpt-5.5, held out from the mechanism/gen
models) guessing the original from the deployed sentence (fill in slot, mechanism disclosed);
hit@k by fuzzy match. Probes scored on the same items:

  P2 appositive-MLM   — cloak.probe.guess_back_risk (v1 frontrunner, working tree)
  P3 multi-mask PLL   — k masks in the slot next to the visible fill; mean P of original tokens
  P4 contrastive      — softmax over {original} u same-type corpus distractors of causal-LM
                        logP(candidate | sentence with fill + disclosure suffix)
  P6 embedding sim    — MiniLM cos(fill, original); surface-metric baseline, expected to lose

Report per probe: AUC(probe score -> attacker hit@1/@5) + mean per-span Spearman across levels
(the tau-walk's actual use case). Arbiter for the fork-1 correlation rule.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/privacy_probe_shootout.py
"""
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from build_arms_artifact import load_artifact  # noqa: E402

from cloak.probe import _pipe, guess_back_risk  # noqa: E402
from inferdpt.llm import LLMClient  # noqa: E402
from inferdpt.pipeline import pmap  # noqa: E402

ATTACKER_MODEL = "gemini-3.1-pro-preview"  # gpt-5.5 upstream is Cloudflare-blocked; opus-4-6 404s
CAUSAL_MODEL = "EleutherAI/pythia-410m"
MAX_ITEMS_PER_CORPUS = 75
N_DISTRACTORS = 15

ATTACK_PROMPT = """One expression in the sentence below was anonymized: "{fill}" replaced a specific original expression of type {typ}.

Sentence: {sent}

Give your 5 best guesses for the original expression, most likely first.
Reply with ONLY a JSON list of 5 strings."""


def _sent_around(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    his = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    return text[lo:min(his) + 1 if his else len(text)].strip()


def collect_items() -> tuple[list[dict], dict]:
    """(span, level) items from the artifact's tau_walk R entries; same-type surface pools."""
    from cloak.corpora import load_task_docs
    art = load_artifact()
    items, pools = [], {}
    for corpus in ("clinical", "enron"):
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
        seen, n = set(), 0
        for doc_id, arms in art[corpus].items():
            for e in arms["tau_walk"][1]:
                if e["action"] != "generalize" or not e.get("lattice"):
                    continue
                pools.setdefault(e["type"], set()).add(e["surface"])
                sent = _sent_around(texts[doc_id], e["start"], e["end"])
                if e["surface"] not in sent:
                    continue
                for lvl in e["lattice"]:
                    key = (e["surface"].lower(), lvl.lower())
                    if key in seen or n >= MAX_ITEMS_PER_CORPUS:
                        continue
                    seen.add(key)
                    n += 1
                    items.append({"corpus": corpus, "span_id": f"{doc_id}:{e['surface']}",
                                  "surface": e["surface"], "type": e["type"], "sent": sent,
                                  "fill": lvl,
                                  "sent_p": sent.replace(e["surface"], lvl)})
    return items, {t: sorted(s) for t, s in pools.items()}


# ---------- ground truth: remote LLM attacker ----------

def attacker_hits(items: list[dict]) -> None:
    from rapidfuzz import fuzz
    llm = LLMClient(ATTACKER_MODEL, temperature=0.0, max_tokens=200)

    def ask(it):  # gemini upstream: 25 req/min -> throttle + retry; cache hits skip the sleep
        import os

        import openai
        from inferdpt.llm import _cache_path
        prompt = ATTACK_PROMPT.format(fill=it["fill"], typ=it["type"], sent=it["sent_p"])
        cp = _cache_path(ATTACKER_MODEL, [{"role": "user", "content": prompt}],
                         {"temperature": 0.0, "max_tokens": 200})
        cached = cp and os.path.exists(cp)
        for _ in range(6):
            try:
                r = llm.generate(prompt)
                if not cached:
                    time.sleep(2.6)
                return r
            except openai.RateLimitError:
                time.sleep(65)
        return ""

    replies = pmap(ask, items, workers=1)
    for it, r in zip(items, replies):
        m = re.search(r"\[.*?\]", r or "", re.DOTALL)
        try:
            guesses = [str(g) for g in json.loads(m.group())][:5] if m else []
        except json.JSONDecodeError:
            guesses = []
        scores = [fuzz.token_sort_ratio(g.lower(), it["surface"].lower()) for g in guesses]
        it["hit1"] = int(bool(scores) and scores[0] >= 85)
        it["hit5"] = int(any(s >= 85 for s in scores))
        it["guesses"] = guesses


# ---------- probes ----------

def p3_pll(items: list[dict]) -> None:
    """Multi-mask PLL: slot = '<mask>*k, {fill},'; score = mean P(original token i at mask i)."""
    fill_pipe = _pipe()
    tok, model = fill_pipe.tokenizer, fill_pipe.model
    for it in items:
        ids = tok(" " + it["surface"], add_special_tokens=False)["input_ids"][:8]
        masks = " ".join([tok.mask_token] * len(ids))
        text = it["sent"].replace(it["surface"], f"{masks}, {it['fill']},")
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero().flatten()
        probs = logits[pos].softmax(-1)
        it["p3"] = float(mean(probs[i, t].item() for i, t in enumerate(ids[:len(pos)]))) \
            if len(pos) else 0.0


_causal = None


def _causal_lm():
    global _causal
    if _causal is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t = AutoTokenizer.from_pretrained(CAUSAL_MODEL)
        m = AutoModelForCausalLM.from_pretrained(CAUSAL_MODEL, torch_dtype=torch.float16)
        m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _causal = (t, m)
    return _causal


def _logp_continuations(prefix: str, conts: list[str]) -> list[float]:
    """Length-normalized logP(cont | prefix) for each continuation, batched."""
    tok, model = _causal_lm()
    pre = tok(prefix)["input_ids"]
    seqs = [pre + tok(" " + c, add_special_tokens=False)["input_ids"] for c in conts]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    batch = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(model.device)
    with torch.no_grad():
        logits = model(batch).logits.log_softmax(-1)
    out = []
    for row, s in zip(range(len(seqs)), seqs):
        lp = [logits[row, i - 1, s[i]].item() for i in range(len(pre), len(s))]
        out.append(mean(lp) if lp else -1e9)
    return out


def p4_contrastive(items: list[dict], pools: dict) -> None:
    """Attacker-knows-mechanism re-identification: softmax over {original} u distractors."""
    import random
    rng = random.Random(0)
    for it in items:
        pool = [s for s in pools.get(it["type"], []) if s.lower() != it["surface"].lower()]
        distractors = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
        cands = [it["surface"]] + distractors
        prefix = (f'{it["sent_p"]}\nThe anonymized phrase "{it["fill"]}" '
                  f'originally read:')
        lps = _logp_continuations(prefix, cands)
        z = torch.tensor(lps).softmax(-1)
        it["p4"] = float(z[0])
        it["p4_n_cands"] = len(cands)


def p6_embed(items: list[dict]) -> None:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    a = m.encode([it["fill"] for it in items], normalize_embeddings=True)
    b = m.encode([it["surface"] for it in items], normalize_embeddings=True)
    for it, x, y in zip(items, a, b):
        it["p6"] = float((x * y).sum())


# ---------- metrics ----------

def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def per_span_rank_agreement(items: list[dict], probe: str) -> float | None:
    """Across levels of one span: does the probe order levels like attacker hit@5?"""
    agree = []
    by_span = {}
    for it in items:
        by_span.setdefault(it["span_id"], []).append(it)
    for its in by_span.values():
        if len(its) < 2:
            continue
        pairs = [(a, b) for a, b in combinations(its, 2) if a["hit5"] != b["hit5"]]
        if not pairs:
            continue
        ok = [(a[probe] > b[probe]) == (a["hit5"] > b["hit5"]) for a, b in pairs
              if a[probe] != b[probe]]
        if ok:
            agree.append(mean(ok))
    return round(mean(agree), 3) if agree else None


def main():
    t0 = time.time()
    items, pools = collect_items()
    print(f"items={len(items)} pools={ {t: len(v) for t, v in pools.items()} }", flush=True)

    attacker_hits(items)
    h1, h5 = mean(i["hit1"] for i in items), mean(i["hit5"] for i in items)
    print(f"attacker {ATTACKER_MODEL}: hit@1={h1:.3f} hit@5={h5:.3f} {time.time()-t0:.0f}s",
          flush=True)

    for it in items:  # P2: working-tree appositive probe
        it["p2"] = guess_back_risk(it["sent_p"], it["surface"], it["fill"])
    print(f"P2 done {time.time()-t0:.0f}s", flush=True)
    p3_pll(items)
    print(f"P3 done {time.time()-t0:.0f}s", flush=True)
    p4_contrastive(items, pools)
    print(f"P4 done {time.time()-t0:.0f}s", flush=True)
    p6_embed(items)
    print(f"P6 done {time.time()-t0:.0f}s", flush=True)

    report = {"attacker": ATTACKER_MODEL, "n_items": len(items),
              "attacker_hit1": round(h1, 3), "attacker_hit5": round(h5, 3), "probes": {}}
    for p in ("p2", "p3", "p4", "p6"):
        report["probes"][p] = {
            "auc_hit1": auc([i[p] for i in items], [i["hit1"] for i in items]),
            "auc_hit5": auc([i[p] for i in items], [i["hit5"] for i in items]),
            "per_span_level_agreement": per_span_rank_agreement(items, p)}
        print(p, report["probes"][p], flush=True)
    report["items"] = [{k: v for k, v in it.items() if k != "sent"} for it in items]
    out = Path("results/privacy_probe_shootout.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
