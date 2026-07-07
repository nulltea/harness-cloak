"""Per-type extractor comparison on echoed spans from extractor_miss_audit jobs.

Denominator excludes true absents: only generalization entries whose replacement was echoed
into out_p as exact, fuzzy90, or band60_90 are counted as present-in-out_p.

Run:
  INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
    .venv/bin/python -u scripts/spikes/extractor_pointer_by_type.py \
    --env data/ranker_env.json --arms data/task_arms_tau0.02.json \
    --corpora clinical --n-docs 16 --workers 6
"""
import argparse
import json
import re
from pathlib import Path

from build_arms_artifact import load_artifact
from train_ranker import assemble, derive_spans, floor_walk_choice

from cloak.corpora import load_task_docs
from cloak.detect import Detector
from cloak.extract import (DETECTOR_POINTER_GLINER_MODEL, FUZZ_MIN, invert,
                           _detect_spans, _dilate_detector_spans, _pointer_assign,
                           _rule_prepass)
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/extractor_pointer_by_type.json")
BAND_LO = 60.0
TYPE_LABELS = {
    "PERSON": "a person's name",
    "ORG": "an organization, company, court or institution",
    "LOC": "a location, address, city or country",
    "DATETIME": "a date, time or duration",
    "CODE": "a reference number or identification code",
    "QUANTITY": "a quantity, amount of money or percentage",
    "DEM": "a demographic attribute (nationality, ethnicity, religion, profession or age)",
    "MISC": "an identifying attribute or event",
}


def classify_echo(fill: str, out_p: str) -> str:
    from rapidfuzz import fuzz
    if re.search(rf"\b{re.escape(fill)}\b", out_p, re.IGNORECASE):
        return "exact"
    al = fuzz.partial_ratio_alignment(fill.lower(), out_p.lower())
    score = al.score if al else 0.0
    if score >= FUZZ_MIN:
        return "fuzzy90"
    if score >= BAND_LO:
        return "band60_90"
    return "absent"


def build_jobs(args) -> tuple[list[dict], list[dict]]:
    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    jobs, metas = [], []
    for corpus in args.corpora.split(","):
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in env["corpora"].get(corpus, {}).items():
            if not d.get("spans") or doc_id not in texts:
                continue
            spans, _ = derive_spans(d["spans"], dict(env["k_floors"]), corpus, "cpu")
            choice = floor_walk_choice(spans)
            doc_p, R = assemble(texts[doc_id], art[corpus][doc_id]["tau_walk"][1],
                                d["spans"], choice)
            jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
            metas.append({"corpus": corpus, "doc_id": doc_id, "R": R})
    return jobs, metas


def hit_rule_echo(cls: str) -> bool:
    return cls in ("exact", "fuzzy90")


def hit_design1(out_p: str, e: dict) -> bool:
    _, stats = invert(out_p, [e])
    return stats["gen_exact"] + stats["gen_fuzzy"] + stats["gen_semantic"] > 0


def pct(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env.json")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json")
    ap.add_argument("--corpora", default="clinical")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tau-det", type=float, default=0.3)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    detector = Detector(gliner_model=DETECTOR_POINTER_GLINER_MODEL)

    rows = {typ: {"type": typ, "description": desc, "present_in_out_p": 0,
                  "rule_extracted": 0, "design1_extracted": 0,
                  "pointer_extracted": 0}
            for typ, desc in TYPE_LABELS.items()}
    examples = []

    for m, o in zip(metas, outs):
        _, _, pointer_residue = _rule_prepass(o["out_p"], m["R"], semantic=False)
        pointer_candidates = _dilate_detector_spans(
            o["out_p"], _detect_spans(detector, o["out_p"],
                                      gliner_model=DETECTOR_POINTER_GLINER_MODEL), args.tau_det)
        pointer_assigned = _pointer_assign(pointer_residue, pointer_candidates,
                                           score_min=0.70, delta=0.05)
        pointer_hit_keys = {(pointer_residue[i]["surface"], pointer_residue[i]["replacement"])
                            for i in pointer_assigned}
        for e in m["R"]:
            if e["action"] != "generalize":
                continue
            typ = e.get("type", "MISC")
            if typ not in rows:
                typ = "MISC"
            cls = classify_echo(e["replacement"], o["out_p"])
            if cls == "absent":
                continue
            rows[typ]["present_in_out_p"] += 1
            rule_hit = hit_rule_echo(cls)
            d1_hit = hit_design1(o["out_p"], e)
            ptr_hit = rule_hit or (e["surface"], e["replacement"]) in pointer_hit_keys
            rows[typ]["rule_extracted"] += int(rule_hit)
            rows[typ]["design1_extracted"] += int(d1_hit)
            rows[typ]["pointer_extracted"] += int(ptr_hit)
            if (d1_hit or ptr_hit) and not rule_hit and len(examples) < 20:
                examples.append({"doc": m["doc_id"], "type": typ, "surface": e["surface"],
                                 "fill": e["replacement"], "echo_class": cls,
                                 "design1_hit": d1_hit, "pointer_hit": ptr_hit})

    out_rows = []
    for typ in TYPE_LABELS:
        row = rows[typ]
        denom = row["present_in_out_p"]
        out_rows.append({**row,
                         "rule_rate": pct(row["rule_extracted"], denom),
                         "design1_rate": pct(row["design1_extracted"], denom),
                         "pointer_rate": pct(row["pointer_extracted"], denom)})

    totals = {"present_in_out_p": sum(r["present_in_out_p"] for r in rows.values()),
              "rule_extracted": sum(r["rule_extracted"] for r in rows.values()),
              "design1_extracted": sum(r["design1_extracted"] for r in rows.values()),
              "pointer_extracted": sum(r["pointer_extracted"] for r in rows.values())}
    totals.update({"rule_rate": pct(totals["rule_extracted"], totals["present_in_out_p"]),
                   "design1_rate": pct(totals["design1_extracted"], totals["present_in_out_p"]),
                   "pointer_rate": pct(totals["pointer_extracted"], totals["present_in_out_p"])})

    report = {"settings": {**vars(args), "detector_gliner_model": DETECTOR_POINTER_GLINER_MODEL},
              "n_docs": len(metas), "rows": out_rows, "totals": totals,
              "non_rule_recovery_examples": examples}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "non_rule_recovery_examples"},
                     indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
