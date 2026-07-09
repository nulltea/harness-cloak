"""Red tests for detector miss-classification — characterize failure modes, not fix them.

Runs a GLiNER detector with the EXACT miner invocation (labels, threshold, batch call from
scripts/build_mined_lattice_profiles.py) over adversarial cases grouped by hypothesized failure
mode (lab-test-as-drug, imaging-as-drug, device-as-drug, abbreviation over-capture, non-medical
capture, OCR junk, body-part-as-condition, assessment-as-entity) plus positive controls
(real drugs/conditions/procedures) and contrast pairs (ace wrap vs ace inhibitor, cad vs
coronary artery disease). Each case runs in two modes: full sentence and bare surface, to
separate context-independent failures from context-induced ones.

A red case "fires" when the model emits a forbidden (surface, runtime-type); a control "passes"
when the expected type is found. High fire rates are the FINDING (empirical honesty) — the
harness reports, it never gates. Re-run after any detector fix to compare; pass
--model data/models/pii_gliner_multidomain/checkpoint-2479 to probe the fine-tuned detector.

    PYTHONPATH=src .venv/bin/python -u scripts/detector_red_tests.py \
        [--model knowledgator/gliner-pii-base-v1.0] [--threshold 0.3] [--out results/...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mined_lattice_profiles import DETECTOR_LABELS, LABEL_TO_RUNTIME_TYPE, _norm

ALL_TYPES = sorted(set(LABEL_TO_RUNTIME_TYPE.values()))
MINED_SPANS = Path("results/mined_lattice_profile_spans.jsonl")

# Each case: category = hypothesized failure mode; target = span under test; wrong = runtime
# types the model must NOT assign the target; right = type a correct model SHOULD assign (controls
# and type-confusion cases where the surface is a real entity of another type).
def _c(category: str, text: str, target: str, wrong: list[str] | None = None, right: str | None = None) -> dict:
    return {"category": category, "text": text, "target": target, "wrong": wrong or [], "right": right}

D, C, P = "drug", "health-condition", "medical-procedure"

CASES: list[dict] = [
    # -- lab tests mined as drug (evidence: a1c cbc bmp psa hcg afp ldh bun pcr hcv-ab) --
    _c("lab-test-as-drug", "Her a1c came back at 7.2 this visit.", "a1c", wrong=[D, C]),
    _c("lab-test-as-drug", "We will order a cbc and a bmp this morning.", "cbc", wrong=[D, C]),
    _c("lab-test-as-drug", "We will order a cbc and a bmp this morning.", "bmp", wrong=[D, C]),
    _c("lab-test-as-drug", "His psa was mildly elevated at 5.1.", "psa", wrong=[D, C]),
    _c("lab-test-as-drug", "Serum hcg was negative on admission.", "hcg", wrong=[D, C]),
    _c("lab-test-as-drug", "The afp level was within normal limits.", "afp", wrong=[D, C]),
    _c("lab-test-as-drug", "Her ldh and bun were both trending down.", "ldh", wrong=[D, C]),
    _c("lab-test-as-drug", "Her ldh and bun were both trending down.", "bun", wrong=[D, C]),
    _c("lab-test-as-drug", "A pcr swab was sent for influenza.", "pcr", wrong=[D, C]),
    _c("lab-test-as-drug", "The hcv antibody test returned negative.", "hcv antibody test", wrong=[D, C]),
    _c("lab-test-as-drug", "Repeat tsh is scheduled in six weeks.", "tsh", wrong=[D, C]),
    _c("lab-test-as-drug", "Serial troponin values were flat.", "troponin", wrong=[D, C]),
    # -- imaging / diagnostics mined as drug (mri ecg ekg emg) --
    _c("imaging-as-drug", "An mri of the lumbar spine was ordered.", "mri", wrong=[D, C]),
    _c("imaging-as-drug", "The ecg showed normal sinus rhythm.", "ecg", wrong=[D, C]),
    _c("imaging-as-drug", "A repeat ekg was unchanged from prior.", "ekg", wrong=[D, C]),
    _c("imaging-as-drug", "An emg confirmed mild carpal tunnel syndrome.", "emg", wrong=[D, C]),
    _c("imaging-as-drug", "A ct scan of the abdomen was unremarkable.", "ct scan", wrong=[D, C]),
    _c("imaging-as-drug", "The chest x ray showed no acute process.", "chest x ray", wrong=[D, C]),
    # -- devices / supplies mined as drug (ace wrap, iud) --
    _c("device-as-drug", "We applied an ace wrap to the left ankle.", "ace wrap", wrong=[D, C, P]),
    _c("device-as-drug", "She had an iud placed two years ago.", "iud", wrong=[D, C]),
    _c("device-as-drug", "He was fitted with a knee brace today.", "knee brace", wrong=[D, C]),
    _c("device-as-drug", "She uses a nebulizer at home as needed.", "nebulizer", wrong=[D, C]),
    _c("device-as-drug", "His insulin pump settings were adjusted.", "insulin pump", wrong=[D, C]),
    _c("device-as-drug", "The pacemaker was interrogated in clinic.", "pacemaker", wrong=[D, C]),
    # -- clinical abbreviations mined as drug (cva cad lad jvd gcs) --
    _c("clinical-abbrev-as-drug", "Past history includes cva in 2019.", "cva", wrong=[D], right=C),
    _c("clinical-abbrev-as-drug", "He has known cad with prior stenting.", "cad", wrong=[D], right=C),
    _c("clinical-abbrev-as-drug", "There is a proximal lad stenosis.", "lad", wrong=[D]),
    _c("clinical-abbrev-as-drug", "No jvd was noted on exam.", "jvd", wrong=[D]),
    _c("clinical-abbrev-as-drug", "gcs was 14 on arrival to the ed.", "gcs", wrong=[D, C]),
    # -- 2-3 letter tokens captured with no disambiguating context (ap bl cda ep gt sh ski dax) --
    _c("short-token-overcapture", "Please see form ap for the full listing.", "ap", wrong=ALL_TYPES),
    _c("short-token-overcapture", "The bl values are recorded in the chart.", "bl", wrong=ALL_TYPES),
    _c("short-token-overcapture", "Refer to section cda of the document.", "cda", wrong=ALL_TYPES),
    _c("short-token-overcapture", "Follow up was arranged at the ep clinic.", "ep", wrong=ALL_TYPES),
    _c("short-token-overcapture", "The gt reading was recorded at bedside.", "gt", wrong=ALL_TYPES),
    _c("short-token-overcapture", "Notes were filed under sh by the clerk.", "sh", wrong=ALL_TYPES),
    _c("short-token-overcapture", "He injured himself on a ski trip last week.", "ski", wrong=ALL_TYPES),
    _c("short-token-overcapture", "The dax entry was left blank on intake.", "dax", wrong=ALL_TYPES),
    # -- fully non-medical spans (durable power of attorney was mined as drug) --
    _c("non-medical-capture", "She completed a durable power of attorney last month.", "durable power of attorney", wrong=ALL_TYPES),
    _c("non-medical-capture", "He updated his living will with his lawyer.", "living will", wrong=ALL_TYPES),
    _c("non-medical-capture", "Please bring your driver's license to the visit.", "driver's license", wrong=ALL_TYPES),
    _c("non-medical-capture", "Her health insurance policy lapsed in March.", "health insurance policy", wrong=ALL_TYPES),
    _c("non-medical-capture", "He signed a new employment contract on Monday.", "employment contract", wrong=ALL_TYPES),
    # -- OCR / paraphrase junk (10 10 20 20; 0 7 millimeter lesion; excipient chemistry) --
    _c("ocr-junk", "Vision was 10 10 20 20 on the chart.", "10 10 20 20", wrong=ALL_TYPES),
    _c("ocr-junk", "Imaging showed a 0 7 millimeter lesion of the liver.", "0 7 millimeter lesion", wrong=ALL_TYPES),
    _c("ocr-junk", "Ingredients include .alpha.-hexylcinnamaldehyde and limonene.", ".alpha.-hexylcinnamaldehyde", wrong=[D]),
    _c("ocr-junk", "Ingredients include .alpha.-hexylcinnamaldehyde and limonene.", "limonene", wrong=[D]),
    # -- body parts mined as health-condition (arm) --
    _c("body-part-as-condition", "He fell directly onto his arm yesterday.", "arm", wrong=[C, D]),
    _c("body-part-as-condition", "She reports soreness around the left knee.", "left knee", wrong=[C, D]),
    _c("body-part-as-condition", "Palpation of the lower back was unremarkable.", "lower back", wrong=[C, D]),
    # -- assessments / generic encounters mined as condition or procedure --
    _c("assessment-as-entity", "She has full active and passive range of motion.", "range of motion", wrong=[C, D]),
    _c("assessment-as-entity", "He is independent in activities of daily living.", "activities of daily living", wrong=[C, D, P]),
    _c("assessment-as-entity", "He presents today for his annual exam.", "annual exam", wrong=[C, D]),
    _c("assessment-as-entity", "A routine blood test was drawn at the visit.", "blood test", wrong=[C, D]),
    _c("assessment-as-entity", "The autoimmune panel was sent to the lab.", "autoimmune panel", wrong=[C, D]),
    # -- positive controls: real entities the detector MUST keep finding (recall guard) --
    _c("control-drug", "She was started on metformin 500 mg twice daily.", "metformin", right=D),
    _c("control-drug", "Lisinopril was increased to 20 mg daily.", "lisinopril", right=D),
    _c("control-drug", "He takes atorvastatin at bedtime.", "atorvastatin", right=D),
    _c("control-drug", "A course of amoxicillin was prescribed.", "amoxicillin", right=D),
    _c("control-drug", "Insulin glargine was titrated up by two units.", "insulin glargine", right=D),
    _c("control-drug", "The mmr vaccine was administered in the left arm.", "mmr vaccine", right=D),
    _c("control-condition", "She has type 2 diabetes and hypertension.", "type 2 diabetes", right=C),
    _c("control-condition", "She has type 2 diabetes and hypertension.", "hypertension", right=C),
    _c("control-condition", "His asthma has been well controlled.", "asthma", right=C),
    _c("control-condition", "He has coronary artery disease with prior stenting.", "coronary artery disease", right=C),
    _c("control-procedure", "She underwent an appendectomy in 2015.", "appendectomy", right=P),
    _c("control-procedure", "A screening colonoscopy is due this year.", "colonoscopy", right=P),
    _c("control-procedure", "He received a blood transfusion post-operatively.", "blood transfusion", right=P),
    _c("control-procedure", "Cardiac catheterization showed two-vessel disease.", "cardiac catheterization", right=P),
    # -- contrast pairs: same token, other sense — sharpest probe of type confusion --
    _c("contrast-pair", "He was started on an ace inhibitor for blood pressure.", "ace inhibitor", right=D),
    _c("contrast-pair", "The pcp prescription was renewed for prophylaxis.", "pcp", right=D),
]


def _contains(pred: str, target: str) -> bool:
    p, t = _norm(pred), _norm(target)
    if not p or not t:
        return False
    return re.search(rf"(^|\s){re.escape(t)}($|\s)", p) is not None or re.search(rf"(^|\s){re.escape(p)}($|\s)", t) is not None


def run_detector(texts: list[str], model: str, threshold: float) -> list[list[dict]]:
    import torch
    from gliner import GLiNER

    from cloak.detect import _install_gliner_bounds_guard

    _install_gliner_bounds_guard()
    gliner = GLiNER.from_pretrained(model)
    if torch.cuda.is_available():
        gliner = gliner.to("cuda")
    preds = gliner.batch_predict_entities(texts, DETECTOR_LABELS, threshold=threshold, batch_size=16)
    out = []
    for ents in preds:
        rows = []
        for ent in ents:
            label = _norm(ent["label"])
            if label in LABEL_TO_RUNTIME_TYPE:
                rows.append({"text": ent["text"], "type": LABEL_TO_RUNTIME_TYPE[label], "score": round(float(ent.get("score", 0.0)), 4)})
        out.append(rows)
    return out


def mined_score_lookup() -> dict[str, list[tuple[str, float]]]:
    """surface -> [(runtime_type, score), ...] from the real mining run, for cross-reference."""
    lookup: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not MINED_SPANS.exists():
        return lookup
    for line in MINED_SPANS.read_text().splitlines():
        row = json.loads(line)
        rt = LABEL_TO_RUNTIME_TYPE.get(_norm(row["detector_label"]))
        if rt:
            lookup[_norm(row["surface"])].append((rt, round(row["score"], 4)))
    return lookup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true", help="assert the span matcher, no GPU")
    # ponytail: default output is the summary table only — per-case fired lines stay in the JSON
    # (dense drug-token dumps in chat trip Fable 5's safeguard fallback; see 2026-07-09 handoff)
    ap.add_argument("--verbose", action="store_true", help="also print per-case fired lines")
    ap.add_argument("--model", default="knowledgator/gliner-pii-base-v1.0")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", default=None, help="results JSON (default derives from model name)")
    args = ap.parse_args()

    if args.selfcheck:
        assert _contains("ace wrap", "ace wrap")                  # exact
        assert _contains("an ace wrap here", "ace wrap")          # pred contains target phrase
        assert _contains("mri", "an mri scan")                    # target contains pred phrase
        assert not _contains("wrapper", "ace wrap")               # substring must not match as word
        assert not _contains("cadence", "cad")                    # abbrev not a substring hit
        assert _contains("cad", "known cad here")
        assert not _contains("", "cad") and not _contains("cad", "")
        print("selfcheck ok")
        return 0

    texts = [c["text"] for c in CASES] + [c["target"] for c in CASES]  # sentence mode + bare mode
    preds = run_detector(texts, args.model, args.threshold)
    sent_preds, bare_preds = preds[: len(CASES)], preds[len(CASES):]
    mined = mined_score_lookup()

    results = []
    for case, sp, bp in zip(CASES, sent_preds, bare_preds):
        row = dict(case)
        for mode, pred in (("sentence", sp), ("bare", bp)):
            hits = [p for p in pred if _contains(p["text"], case["target"])]
            row[f"{mode}_preds"] = pred
            row[f"{mode}_wrong_hits"] = [h for h in hits if h["type"] in case["wrong"]]
            row[f"{mode}_right_hit"] = next((h for h in hits if case["right"] and h["type"] == case["right"]), None)
        row["mined_hits"] = mined.get(_norm(case["target"]), [])
        results.append(row)

    # summary per category
    summary: dict[str, dict] = {}
    for row in results:
        s = summary.setdefault(row["category"], {"cases": 0, "fired_sentence": 0, "fired_bare": 0,
                                                 "control_hit": 0, "controls": 0, "wrong_scores": []})
        s["cases"] += 1
        s["fired_sentence"] += bool(row["sentence_wrong_hits"])
        s["fired_bare"] += bool(row["bare_wrong_hits"])
        s["wrong_scores"] += [h["score"] for h in row["sentence_wrong_hits"]]
        if row["right"]:
            s["controls"] += 1
            s["control_hit"] += bool(row["sentence_right_hit"])
    for s in summary.values():
        scores = s.pop("wrong_scores")
        s["wrong_score_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        s["wrong_score_max"] = max(scores) if scores else None

    out_path = Path(args.out) if args.out else Path("results/detector_red_tests") / (re.sub(r"[^\w.-]+", "-", args.model) + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "threshold": args.threshold,
                                    "summary": summary, "results": results}, indent=2, sort_keys=True) + "\n")

    print(f"\nmodel={args.model} threshold={args.threshold}\n")
    print(f"{'category':<28}{'cases':>6}{'fired-sent':>11}{'fired-bare':>11}{'ctrl-recall':>12}{'wrong-score μ/max':>19}")
    for cat, s in sorted(summary.items()):
        ctrl = f"{s['control_hit']}/{s['controls']}" if s["controls"] else "-"
        ws = f"{s['wrong_score_mean']}/{s['wrong_score_max']}" if s["wrong_score_mean"] is not None else "-"
        print(f"{cat:<28}{s['cases']:>6}{s['fired_sentence']:>11}{s['fired_bare']:>11}{ctrl:>12}{ws:>19}")
    fired = [r for r in results if r["sentence_wrong_hits"] or r["bare_wrong_hits"]]
    print(f"\nred cases fired: {len(fired)}/{sum(1 for c in CASES if c['wrong'])} | detail: {out_path}")
    if args.verbose:
        for r in fired:
            wrong = r["sentence_wrong_hits"] or r["bare_wrong_hits"]
            kinds = ", ".join(sorted({f"{h['type']}@{h['score']}" for h in wrong}))
            mode = "sent" if r["sentence_wrong_hits"] else "bare"
            print(f"  [{r['category']}/{mode}] {r['target']!r} -> {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
