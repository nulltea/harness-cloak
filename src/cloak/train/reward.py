"""Surrogate utility for substitutor training: a local mini round trip, no generation model.

The task corpora are selected so the gold output restates substituted spans
(docs/specs/benchmarks.md), which makes two model-free signals price a candidate (doc_p, R):

- U_QA (answerability): for each restated span (R surface found in gold), a cloze question
  built from the gold sentence is answered from doc_p by an extractive reader; the answer is
  inverted through R (rule extractor, `extract.invert`) and token-F1-scored against the original
  surface. Invertible coarsening scores 1.0 by construction; destruction (unanswerable /
  unalignable) scores 0 — so utility cannot reward under-anonymization.
- U_NLI (premise retention): each gold sentence, R-generalized, must be entailed by the
  best-overlap window of doc_p (same NLI model as the lattice truthfulness gate).

Plan: docs/plans/2026-07-02-surrogate-grpo-training.md.
"""
import re

from cloak.extract import invert
from cloak.lattice import NLI_MODEL

QA_MODEL = "deepset/roberta-base-squad2"
NLI_THRESH = 0.5
_qa = None
_nli = None


def _qa_models():
    # transformers 5.x dropped the question-answering pipeline; run the reader manually
    global _qa
    if _qa is None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QA_MODEL)
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _qa = (tok, model)
    return _qa


def _qa_answer(question: str, context: str, max_answer_toks: int = 30) -> str:
    """Best extractive answer, or '' when the no-answer (CLS) span wins (SQuAD2 semantics)."""
    import torch
    tok, model = _qa_models()
    enc = tok(question, context, return_tensors="pt", truncation="only_second",
              max_length=384, stride=128, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding=True)
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inputs)
    best, best_score = "", -1e9
    for i in range(inputs["input_ids"].shape[0]):
        s, e = out.start_logits[i], out.end_logits[i]
        null_score = (s[0] + e[0]).item()
        ctx_mask = torch.tensor([sid != 1 for sid in enc.sequence_ids(i)], device=s.device)
        s = s.masked_fill(ctx_mask, -1e9)
        e = e.masked_fill(ctx_mask, -1e9)
        si = int(s.argmax())
        ei = si + int(e[si:si + max_answer_toks].argmax())
        score = (s[si] + e[ei]).item()
        if score > null_score and score > best_score:
            lo, hi = offsets[i][si], offsets[i][ei]
            best, best_score = context[lo[0]:hi[1]], score
    return best


def _nli_pipe():
    global _nli
    if _nli is None:
        import torch
        from transformers import pipeline
        _nli = pipeline("text-classification", model=NLI_MODEL,
                        device=0 if torch.cuda.is_available() else -1)
    return _nli


def _sentences(text: str) -> list[str]:
    # title abbreviations must not end a sentence ("followed by Dr. Kumar")
    parts = re.split(r"(?<=[.!?])(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!Mrs\.)\s+|\n+", text,
                     flags=re.IGNORECASE)
    return [s.strip() for s in parts if s and len(s.strip()) > 20]


def _toks(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def restated_probes(R: list[dict], gold: str) -> list[dict]:
    """R entries whose original surface the gold output restates, with the gold sentence.

    Exact word-boundary match, then fuzzy (rapidfuzz partial ratio) for surfaces >= 5
    chars — gold restates paraphrased ("50 years old" vs "50-year-old"); exact-only
    measured 0.25-1 probes/doc vs the 5-10 the reward needs (2026-07-02 gate run).
    """
    # No role filter here: _is_role_phrase is a lowercase-PERSON heuristic (its [a-z]+
    # tokenization turns "November 7" into the WordNet noun "november") and ate ~60% of
    # unique surfaces (measured 2026-07-03, 0.88 probes/doc on clinical). Gold restatement
    # is the relevance filter; the teacher validity check drops no-fact role probes.
    from rapidfuzz import fuzz

    def canon(t: str) -> str:
        # spoken-vs-written variants seen in the corpora: "doctor kumar"/"Dr. Kumar",
        # "40 milligrams"/"40 mg". ponytail: spelled-out numbers/dates ("July thirty
        # first") stay unmatched; add a number normalizer if probe supply still short.
        t = re.sub(r"\bdr\.?(?=\s)", "doctor", t.lower())
        return re.sub(r"\bmilligrams?\b", "mg", t)

    sents = _sentences(gold)
    probes, seen = [], set()
    for e in R:
        key = canon(e["surface"])
        if key in seen:  # dedup mentions
            continue
        pat = re.compile(rf"\b{re.escape(key)}\b")
        sent = next((s for s in sents if pat.search(canon(s))), None)
        if sent is None and len(e["surface"]) >= 5:
            scored = [(fuzz.partial_ratio(key, canon(s)), s) for s in sents]
            best = max(scored, default=(0, None))
            if best[0] >= 85:
                sent = best[1]
        if sent is None:
            continue
        seen.add(key)
        probes.append({"entry": e, "gold_sent": sent})
    return probes


def u_qa(doc_p: str, R: list[dict], probes: list[dict]) -> tuple[float | None, list[dict]]:
    """Answerability of restated-span probes from doc_p, scored after R-inversion.

    Each probe: {"surface": original span text, "question": natural question whose answer
    is that span}. Questions are built once per document (train/probes.py — teacher-written,
    cached; cloze phrasings are out-of-distribution for SQuAD2 readers and abstain).
    Returns (mean F1 | None if no probes, per-probe details).
    """
    if not probes:
        return None, []
    details = []
    for p in probes:
        # the question must live in doc_p's generalized space: teacher questions quote
        # other spans' specific surfaces ("Martha Collins"), ungroundable in doc_p
        answer = _qa_answer(generalize_text(p["question"], R), doc_p)
        inv_ans, _ = invert(answer, R) if answer else ("", None)
        f1 = token_f1(inv_ans, p["surface"])
        details.append({"surface": p["surface"], "answer": answer,
                        "inverted": inv_ans, "f1": round(f1, 3)})
    return sum(d["f1"] for d in details) / len(details), details


def fact_recall(out_final: str, probes: list[dict]) -> float | None:
    """Realized utility ground truth: do the gold-restated facts survive the round trip?

    Same probes as u_qa, but the reader answers from out_final (already inverted, original
    space — no question generalization, no answer inversion). Mean token-F1 vs the original
    surface; None when the doc has no probes.
    """
    if not probes:
        return None
    return sum(token_f1(_qa_answer(p["question"], out_final), p["surface"])
               for p in probes) / len(probes)


def generalize_text(text: str, R: list[dict]) -> str:
    """Apply R's surface->replacement map to arbitrary text (longest surface first)."""
    for e in sorted(R, key=lambda e: -len(e["surface"])):
        text = re.sub(rf"\b{re.escape(e['surface'])}\b", e["replacement"], text,
                      flags=re.IGNORECASE)
    return text


def _best_window(prop: str, doc_sents: list[str], width: int = 1) -> str:
    """doc_p sentence with highest token overlap to the proposition, plus neighbors.

    ponytail: lexical-overlap retrieval; embedding retrieval if NLI premises miss.
    """
    pt = set(_toks(prop))
    scores = [len(pt & set(_toks(s))) for s in doc_sents]
    i = max(range(len(scores)), key=scores.__getitem__)
    return " ".join(doc_sents[max(0, i - width):i + width + 1])


def u_nli(doc_p: str, R: list[dict], gold: str) -> tuple[float | None, list[dict]]:
    """Fraction of R-generalized gold sentences entailed by their best doc_p window."""
    props = [generalize_text(s, R) for s in _sentences(gold)]
    doc_sents = _sentences(doc_p) or [doc_p]
    if not props:
        return None, []
    pairs = [{"text": _best_window(p, doc_sents), "text_pair": p} for p in props]
    outs = _nli_pipe()(pairs, top_k=None, truncation=True)
    details = []
    for p, scores in zip(props, outs):
        ent = next(d["score"] for d in scores if d["label"] == "entailment")
        details.append({"prop": p[:80], "entailment": round(ent, 3)})
    frac = sum(d["entailment"] >= NLI_THRESH for d in details) / len(details)
    return frac, details


def stage1_reward(A: float, U: float | None, alpha: float) -> float:
    """The normative stage-1 reward (spec §5): r = α(1−A) + (1−α)·u_qa.

    A = mean P6 fill-proximity over level-mode fills (cloak.probe.reward_privacy, or the
    environment artifact's cached p6 values — identical numbers, fill_proximity is
    deterministic). U = u_qa over the doc's TRAIN-split probes; None (no probes) is a
    caller error for training docs — train only on probe-bearing docs.
    """
    return alpha * (1.0 - A) + (1.0 - alpha) * (U if U is not None else 0.0)


def u_surr(doc_p: str, R: list[dict], gold: str, probes: list[dict] | None = None) -> dict:
    """LEGACY diagnostic (gate history): mean of u_qa and u_nli. NOT the training reward —
    u_nli is off the normative path (measured: it degrades gate agreement 0.367→0.183);
    stage-1 uses stage1_reward above."""
    qa, qa_det = u_qa(doc_p, R, probes or [])
    nli, nli_det = u_nli(doc_p, R, gold)
    parts = [x for x in (qa, nli) if x is not None]
    return {"u_qa": qa, "u_nli": nli,
            "u_surr": sum(parts) / len(parts) if parts else None,
            "n_probes": len(qa_det), "n_props": len(nli_det),
            "qa_details": qa_det, "nli_details": nli_det}


if __name__ == "__main__":
    R = [
        {"action": "placeholder", "surface": "Sarah Johnson", "replacement": "<PERSON_1>"},
        {"action": "generalize", "surface": "34", "replacement": "thirty-something"},
        {"action": "generalize", "surface": "Oslo", "replacement": "a Norwegian city"},
        {"action": "generalize", "surface": "12 March 2019", "replacement": "early 2019"},
    ]
    gold = ("Sarah Johnson, 34, a nurse from Oslo, presented on 12 March 2019 with chest pain. "
            "She was prescribed aspirin and advised rest.")
    doc_good = ("<PERSON_1> is thirty-something and works as a nurse in a Norwegian city. "
                "She came in early 2019 complaining of chest pain. "
                "She got aspirin and was told to rest.")
    doc_destroyed = "<PERSON_1> is a person. An event occurred at some point."
    probes = [  # teacher-written in production (train/probes.py); handwritten here
        {"surface": "Sarah Johnson", "question": "What is the name of the patient?"},
        {"surface": "34", "question": "How old is the nurse?"},
        {"surface": "Oslo", "question": "Where does the nurse work?"},
        {"surface": "12 March 2019", "question": "When did she come in with chest pain?"},
    ]

    good = u_surr(doc_good, R, gold, probes)
    bad = u_surr(doc_destroyed, R, gold, probes)
    print("good:", {k: good[k] for k in ("u_qa", "u_nli", "u_surr", "n_probes", "n_props")})
    for d in good["qa_details"]:
        print("  ", d)
    print("destroyed:", {k: bad[k] for k in ("u_qa", "u_nli", "u_surr")})

    assert good["n_probes"] == 4, good["n_probes"]  # all four surfaces restated in gold
    assert good["u_surr"] is not None and bad["u_surr"] is not None
    assert good["u_surr"] > bad["u_surr"] + 0.2, (good["u_surr"], bad["u_surr"])
    assert token_f1("Sarah Johnson", "sarah johnson") == 1.0
    assert token_f1("", "Oslo") == 0.0
    print("reward.py self-check OK")
