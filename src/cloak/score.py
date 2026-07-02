"""Reference-scored utility for task-oriented eval: ROUGE-L (cheap) + optional BERTScore.

Both take multi-reference lists (max over refs for ROUGE; BERTScore's own multi-ref).
Spec: docs/specs/benchmarks.md.
"""
_scorer = None


def _rl_scorer():
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _scorer


def rouge_l(pred: str, refs: list[str]) -> float:
    s = _rl_scorer()
    return max(s.score(r, pred)["rougeL"].fmeasure for r in refs)


def bertscore_f1(preds: list[str], refs_list: list[list[str]], lang: str = "en") -> list[float]:
    from bert_score import score
    _, _, f = score(preds, refs_list, lang=lang, verbose=False)
    return f.tolist()


def score_batch(preds: list[str], refs_list: list[list[str]],
                use_bertscore: bool = False) -> dict:
    """Per-item scores. refs_list[i] = reference(s) for preds[i]."""
    out = {"rougeL": [rouge_l(p, rs) for p, rs in zip(preds, refs_list)]}
    if use_bertscore:
        out["bertscore_f1"] = bertscore_f1(preds, refs_list)
    return out


if __name__ == "__main__":
    preds = ["the patient has diabetes and depression", "review title policy easements"]
    refs = [["patient diagnosed with diabetes and depression"],
            ["need to review your title policy's easements section", "title policy question"]]
    r = score_batch(preds, refs)
    assert len(r["rougeL"]) == 2 and all(0 <= x <= 1 for x in r["rougeL"]), r
    assert r["rougeL"][0] > 0.5, r  # strong overlap
    print("rougeL:", [round(x, 3) for x in r["rougeL"]])
    print("score.py self-check OK")
