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
# u_gold scorer LM (spec §5.1): pinned + frozen for the whole gate->train->eval cycle.
# Selection: Qwen/Qwen2.5-1.5B-Instruct when reachable (HF cache/network), else the local
# EleutherAI/pythia-410m fallback. The value below is THE pin recorded in the gate/training
# records; edit it only with a re-gate (a scorer change invalidates trained policies).
GOLD_SCORER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NLI_THRESH = 0.5
_qa = None
_nli = None
_gold = None


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


def _gold_model():
    """Lazy module-level cache for the pinned u_gold scorer LM (spec §5.1). fp16 on GPU."""
    global _gold
    if _gold is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GOLD_SCORER_MODEL)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            GOLD_SCORER_MODEL,
            dtype=torch.float16 if dev == "cuda" else torch.float32)
        model.to(dev).eval()
        _gold = (tok, model)
    return _gold


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


def canon(t: str) -> str:
    """Canonicalize for restatement matching: spoken-vs-written variants seen in the
    corpora ("doctor kumar"/"Dr. Kumar", "40 milligrams"/"40 mg"). ponytail: spelled-out
    numbers/dates ("July thirty first") stay unmatched; add a number normalizer if probe
    supply still short."""
    t = re.sub(r"\bdr\.?(?=\s)", "doctor", t.lower())
    return re.sub(r"\bmilligrams?\b", "mg", t)


def _match_gold_sentence(surface: str, sents: list[str]) -> str | None:
    """The gold sentence that restates `surface`: canonicalized exact word-boundary match,
    then rapidfuzz partial ratio >= 85 for surfaces >= 5 chars. Shared by restated_probes
    (utility probes) and gold_fact_spans (u_gold fact masks) — one matching rule."""
    from rapidfuzz import fuzz
    key = canon(surface)
    pat = re.compile(rf"\b{re.escape(key)}\b")
    sent = next((s for s in sents if pat.search(canon(s))), None)
    if sent is None and len(surface) >= 5:
        best = max(((fuzz.partial_ratio(key, canon(s)), s) for s in sents), default=(0, None))
        if best[0] >= 85:
            sent = best[1]
    return sent


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
    sents = _sentences(gold)
    probes, seen = [], set()
    for e in R:
        key = canon(e["surface"])
        if key in seen:  # dedup mentions
            continue
        sent = _match_gold_sentence(e["surface"], sents)
        if sent is None:
            continue
        seen.add(key)
        probes.append({"entry": e, "gold_sent": sent})
    return probes


def gold_fact_spans(doc_gold: str, R_walk: list[dict]) -> list[dict]:
    """Fact spans for u_gold (spec §5.1/§4.2): unique substituted surfaces the gold output
    restates, as char offsets into `doc_gold`. Restatement is decided by the SAME matcher
    restated_probes uses (_match_gold_sentence); offsets are then located raw (word-boundary,
    else rapidfuzz alignment >= 85) so the returned span is a real substring of doc_gold.

    Each item: {"surface", "gold_start", "gold_end"}. Dedup by canon key (repeat mentions
    map to their first restatement)."""
    from rapidfuzz import fuzz
    sents = _sentences(doc_gold)
    facts, seen = [], set()
    for e in R_walk:
        key = canon(e["surface"])
        if key in seen:
            continue
        sent = _match_gold_sentence(e["surface"], sents)
        if sent is None:
            continue
        surface = e["surface"]
        # restrict the offset search to the matched sentence's region in doc_gold
        base = doc_gold.find(sent)
        region, off = (sent, base) if base >= 0 else (doc_gold, 0)
        m = re.search(rf"\b{re.escape(surface)}\b", region, re.IGNORECASE)
        if m:
            start, end = off + m.start(), off + m.end()
        elif len(surface) >= 5:
            al = fuzz.partial_ratio_alignment(surface.lower(), region.lower())
            if al is None or al.score < 85:
                continue
            start, end = off + al.dest_start, off + al.dest_end
        else:
            continue
        if start >= end:
            continue
        seen.add(key)
        facts.append({"surface": surface, "gold_start": start, "gold_end": end})
    return facts


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


def _score_facts(doc_p: str, R: list[dict], gold: str, facts: list[dict],
                 template: str | None = None) -> tuple[float, list[dict]]:
    """Mean teacher-forced fact-token log-prob under the pinned scorer LM (spec §5.1).

    Per fact j (anti-leak, per-fact cleaned prefix): context = task-prompt(doc_p) +
    generalize_text(gold[:gold_start_j], R) — which R-generalizes every OTHER fact surface
    AND replaces earlier mentions of fact j's own surface with its doc_p replacement (both
    are just "surface -> R replacement"); the scored continuation is the RAW gold fact
    tokens gold[gold_start_j:gold_end_j]. One batched (right-padded) forward across facts;
    returns (mean over facts of per-fact mean-token-logprob, per-fact rows)."""
    import torch
    from cloak.tasks import CLINICAL_NOTE
    tok, model = _gold_model()
    prompt = (template or CLINICAL_NOTE).format(doc=doc_p)
    ctx_head = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
    metas = []  # (fact_idx, full_ids, ctx_len, cont_len)
    for i, f in enumerate(facts):
        prefix = generalize_text(gold[:f["gold_start"]], R)
        cont = gold[f["gold_start"]:f["gold_end"]]
        ctx_ids = tok(ctx_head + prefix, add_special_tokens=False).input_ids
        full_ids = tok(ctx_head + prefix + cont, add_special_tokens=False).input_ids
        # boundary-safe continuation: score from where the two tokenizations diverge, so a
        # subword merge across the prefix/cont seam is scored on the true full-string tokens
        k = 0
        while k < len(ctx_ids) and k < len(full_ids) and ctx_ids[k] == full_ids[k]:
            k += 1
        if len(full_ids) - k <= 0:
            continue
        metas.append((i, full_ids, k, len(full_ids) - k))
    if not metas:
        return 0.0, []
    maxlen = max(len(m[1]) for m in metas)
    pad_id = tok.pad_token_id
    input_ids, attn = [], []
    for _, ids, _, _ in metas:  # RIGHT-pad: real tokens lead, causal mask never sees pad
        p = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * p)
        attn.append([1] * len(ids) + [0] * p)
    input_ids = torch.tensor(input_ids, device=model.device)
    attn = torch.tensor(attn, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    per_fact = []
    for row, (fi, ids, ctx_len, cont_len) in enumerate(metas):
        tgt = input_ids[row, ctx_len:ctx_len + cont_len]
        pred = logp[row, ctx_len - 1:ctx_len + cont_len - 1, :]  # logits[t] predicts token t+1
        tlp = pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        per_fact.append({"surface": facts[fi]["surface"], "score": tlp.mean().item()})
    return sum(p["score"] for p in per_fact) / len(per_fact), per_fact


def u_gold(doc_p: str, R: list[dict], gold: str, facts: list[dict],
           anchors: dict | None = None, template: str | None = None
           ) -> tuple[float | None, dict]:
    """Gold-conditional fact-likelihood utility (spec §5.1): the per-fact anti-leak
    teacher-forced score of doc_p, clipped-normalized between per-doc anchors
    (U_lo all-placeholder floor, U_hi doc_orig ceiling; from u_gold_anchors).

    Returns (u_gold in [0, 1] | None if excluded, details). Edge cases (normative, §5.1):
    empty fact mask, or |U_hi - U_lo| < 0.05 nats, -> (None, {"excluded": reason})."""
    if not facts:
        return None, {"excluded": "empty_facts"}
    if anchors is None:
        raise ValueError("u_gold needs anchors={'U_hi','U_lo'} — call u_gold_anchors first")
    U_hi, U_lo = anchors["U_hi"], anchors["U_lo"]
    if U_hi is None or U_lo is None or abs(U_hi - U_lo) < 0.05:
        return None, {"excluded": "anchor_sep<0.05", "U_hi": U_hi, "U_lo": U_lo}
    raw, per_fact = _score_facts(doc_p, R, gold, facts, template)
    norm = min(1.0, max(0.0, (raw - U_lo) / (U_hi - U_lo)))
    return norm, {"raw": raw, "U_hi": U_hi, "U_lo": U_lo, "per_fact": per_fact}


def u_gold_anchors(doc: str, facts: list[dict], art_entry: dict,
                   template: str | None = None) -> dict:
    """Per-doc u_gold anchors (spec §5.1): U_hi = score on doc_orig (empty R, nothing
    hidden), U_lo = score on the all-placeholder assembly (everything hidden).

    art_entry is the per-doc bag the caller composes: {"gold", "spans", "tau_walk"} — gold
    output text, decision spans (env), and the artifact's tau_walk [doc_p, R] whose R is the
    placeholder-token template. The all-placeholder doc_p is built the way reward_gate.py's
    all_placeholder arm does (assemble with each span's terminal placeholder action)."""
    if not facts:
        return {"U_hi": None, "U_lo": None}
    gold = art_entry["gold"]
    U_hi, _ = _score_facts(doc, [], gold, facts, template)
    from train_ranker import assemble  # scripts/ on path at gate/train time (PYTHONPATH=src:scripts)
    spans = art_entry["spans"]
    R_walk = art_entry["tau_walk"][1]
    ph_choice = {s["surface"].lower(): s["actions"][-1] for s in spans}
    doc_lo, R_lo = assemble(doc, R_walk, spans, ph_choice)[:2]
    U_lo, _ = _score_facts(doc_lo, R_lo, gold, facts, template)
    return {"U_hi": U_hi, "U_lo": U_lo}


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

    # ---- u_gold scorer self-check (spec §5.1) ----
    print(f"u_gold scorer LM: {GOLD_SCORER_MODEL}")
    facts = gold_fact_spans(gold, R)
    assert {f["surface"] for f in facts} == {e["surface"] for e in R}, facts
    fact_34 = next(f for f in facts if f["surface"] == "34")
    assert gold[fact_34["gold_start"]:fact_34["gold_end"]] == "34", fact_34

    kept = {"Sarah Johnson": "Sarah Johnson", "Oslo": "Oslo", "12 March 2019": "12 March 2019"}

    def variant(age_fill):  # doc_p + R differing ONLY in the age treatment
        doc = (f"Sarah Johnson, {age_fill}, a nurse from Oslo, came in on 12 March 2019 "
               "with chest pain.")
        Rv = [{"surface": s, "replacement": r, "action": "generalize"}
              for s, r in kept.items()]
        Rv.append({"surface": "34", "replacement": age_fill, "action": "generalize"})
        return doc, Rv

    # (a) ordering on the "34" fact, RAW per-fact scores (keep > coarse fill > all-placeholder)
    s_keep = _score_facts(*variant("34"), gold, [fact_34])[1][0]["score"]
    s_coarse = _score_facts(*variant("thirty-something"), gold, [fact_34])[1][0]["score"]
    s_ph = _score_facts(*variant("<QUANTITY_1>"), gold, [fact_34])[1][0]["score"]
    print(f"(a) 34-fact raw logprob  keep={s_keep:.3f} coarse={s_coarse:.3f} ph={s_ph:.3f}")
    assert s_keep > s_coarse > s_ph, (s_keep, s_coarse, s_ph)

    # (b) normalized u_gold in [0,1]; doc_orig ~= 1, all-placeholder ~= 0
    doc_orig = ("Sarah Johnson, 34, a nurse from Oslo, came in on 12 March 2019 "
                "with chest pain.")
    all_ph_R = [{"surface": "Sarah Johnson", "replacement": "<PERSON_1>", "action": "placeholder"},
                {"surface": "34", "replacement": "<QUANTITY_1>", "action": "placeholder"},
                {"surface": "Oslo", "replacement": "<LOC_1>", "action": "placeholder"},
                {"surface": "12 March 2019", "replacement": "<DATETIME_1>", "action": "placeholder"}]
    all_ph_doc = generalize_text(doc_orig, all_ph_R)
    U_hi, _ = _score_facts(doc_orig, [], gold, facts)
    U_lo, _ = _score_facts(all_ph_doc, all_ph_R, gold, facts)
    anchors = {"U_hi": U_hi, "U_lo": U_lo}
    print(f"(b) anchors U_hi={U_hi:.3f} U_lo={U_lo:.3f}")
    assert U_hi - U_lo >= 0.05, anchors  # real separation, not an excluded doc
    u_hi_norm, _ = u_gold(doc_orig, [], gold, facts, anchors)
    u_lo_norm, _ = u_gold(all_ph_doc, all_ph_R, gold, facts, anchors)
    print(f"(b) u_gold doc_orig={u_hi_norm} all_placeholder={u_lo_norm}")
    assert u_hi_norm == 1.0 and u_lo_norm == 0.0, (u_hi_norm, u_lo_norm)
    u_mid, _ = u_gold(*variant("thirty-something"), gold, facts, anchors)
    assert 0.0 <= u_mid <= 1.0, u_mid

    # (c) anti-leak: an earlier raw mention of the fact in the gold prefix must NOT inflate
    # the score — the self-mention replacement rule (R applied to the prefix) closes it.
    gold_dup = "Sarah Johnson is 34. The record notes her age is 34 again here."
    second = gold_dup.rfind("34")
    fact_dup = {"surface": "34", "gold_start": second, "gold_end": second + 2}
    ph_doc, _ = variant("<QUANTITY_1>")
    R_leak = [{"surface": s, "replacement": r, "action": "generalize"} for s, r in kept.items()]
    R_clean = R_leak + [{"surface": "34", "replacement": "<QUANTITY_1>", "action": "placeholder"}]
    s_leak = _score_facts(ph_doc, R_leak, gold_dup, [fact_dup])[1][0]["score"]
    s_clean = _score_facts(ph_doc, R_clean, gold_dup, [fact_dup])[1][0]["score"]
    print(f"(c) anti-leak  leaky-prefix={s_leak:.3f} cleaned-prefix={s_clean:.3f}")
    assert s_clean < s_leak, (s_clean, s_leak)  # replacement removes the copy cue

    # (d) exclusions -> None
    assert u_gold(doc_orig, [], gold, [], anchors)[0] is None
    assert u_gold(doc_orig, [], gold, facts, {"U_hi": 1.0, "U_lo": 0.98})[0] is None
    print("u_gold self-check OK")

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
