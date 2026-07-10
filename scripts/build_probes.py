"""Validated probe build (spec Phase 0 step 4): teacher questions + anchor validation.

Per doc: candidate probes from the gemma teacher (cloak.train.probes, cached) -> two anchor
round trips through the PINNED reward model (ceiling = doc_orig, floor = all-placeholder,
both full round trips incl. inversion) -> keep iff ceiling f1 >= TH and floor f1 < TH.
The floor check drops probes the all-placeholder baseline already answers (echoed
placeholders invert perfectly — such probes have no dynamic range above the safest action).

Writes data/probes_validated.json + results/probe_health.json. Docs with < 3 surviving
train probes are listed in excluded_docs (spec: excluded from the RL reward, never
silently kept).

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/build_probes.py [--corpora clinical,enron,aeslc]
       [--n-docs 16] [--workers 8] [--th 0.5] [--seed 0]
"""
import argparse
import datetime
import json
import random
from pathlib import Path

TH = 0.5
OUT = Path("data/probes_validated.json")
REPORT = Path("results/probe_health.json")


def validate_probes(cands, hi_f1s, lo_f1s, th=TH):
    """Pure keep/drop: probe survives iff answerable at the ceiling anchor AND not already
    answered at the floor anchor. Returns (kept, rejected_ceiling, rejected_floor)."""
    kept, rej_c, rej_f = [], [], []
    for p, hi, lo in zip(cands, hi_f1s, lo_f1s):
        if hi < th:
            rej_c.append(p)
        elif lo >= th:
            rej_f.append(p)
        else:
            kept.append(p)
    return kept, rej_c, rej_f


def split_by_fact(kept, seed=0):
    """Train/heldout split at FACT granularity. Kept questions are grouped by canon(surface);
    ALL questions of a fact travel together (fact leakage across splits would corrupt the
    heldout read-out). Facts are shuffled (seeded); hold out max(1, n_facts // 4) facts when
    n_facts >= 2. Returns (train_questions, heldout_questions, n_train_facts)."""
    from cloak.train.reward import canon
    facts = {}
    for p in kept:
        facts.setdefault(canon(p["surface"]), []).append(p)
    keys = list(facts)
    random.Random(seed).shuffle(keys)
    n_hold = max(1, len(keys) // 4) if len(keys) >= 2 else 0
    train = [p for k in keys[n_hold:] for p in facts[k]]
    heldout = [p for k in keys[:n_hold] for p in facts[k]]
    return train, heldout, len(keys) - n_hold


def ladder_health_row(*, docs, spans, rung_candidates, rung_kept, decisions_kept):
    return {
        "docs": docs,
        "spans": spans,
        "rung_candidates": rung_candidates,
        "rung_kept": rung_kept,
        "reader_rung_reject_rate": round(
            (rung_candidates - rung_kept) / max(rung_candidates, 1), 3
        ),
        "tiers_per_span_kept": round(rung_kept / max(spans, 1), 2),
        "decisions_kept": decisions_kept,
        "decisions_kept_per_doc": round(decisions_kept / max(docs, 1), 2),
    }


def _span_rungs(span):
    levels = [a["fill"] for a in span.get("actions", []) if a.get("mode") == "level"]
    return [span["surface"], *levels] if levels else []


def _validated_rung0_lookup(path=OUT):
    from cloak.train.reward import canon

    if not path.exists():
        return {}
    artifact = json.loads(path.read_text())
    docs = artifact.get("docs", {}) if isinstance(artifact, dict) else {}
    lookup = {}
    for doc_id, payload in docs.items():
        for split in ("train", "heldout"):
            for probe in payload.get(split, []):
                q = probe.get("question") or probe.get("q")
                surface = probe.get("surface")
                if q and surface:
                    lookup.setdefault(doc_id, {})[canon(surface)] = probe
    return lookup


def _with_validated_rung0(entries, spans, validated):
    from cloak.train.reward import canon

    by_surface = {canon(s["surface"]): _span_rungs(s) for s in spans if _span_rungs(s)}
    out = []
    replaced = set()
    for e in entries:
        key = canon(e.get("surface", ""))
        if key not in by_surface:
            out.append(e)
            continue
        row = {**e, "rungs": e.get("rungs") or by_surface[key]}
        if row.get("rung") == 0 and key in validated:
            if key not in replaced:
                probe = validated[key]
                out.append({
                    "surface": row["surface"],
                    "rung": 0,
                    "q": probe.get("question") or probe.get("q"),
                    "a": row["surface"],
                    "rungs": by_surface[key],
                    "source": "probes_validated",
                })
                replaced.add(key)
            continue
        out.append(row)
    present_r0 = {(canon(e.get("surface", "")), e.get("rung")) for e in out}
    for s in spans:
        key = canon(s["surface"])
        rungs = by_surface.get(key)
        if rungs and key in validated and (key, 0) not in present_r0:
            probe = validated[key]
            out.append({
                "surface": s["surface"],
                "rung": 0,
                "q": probe.get("question") or probe.get("q"),
                "a": s["surface"],
                "rungs": rungs,
                "source": "probes_validated",
            })
    return out


def _validated_entries(entries, rows):
    out = []
    for e, r in zip(entries, rows):
        row = {**e, "validation": r, "kept": r["verdict"] == "kept"}
        if "span_ids" in r:
            row["span_ids"] = r["span_ids"]
        out.append(row)
    return out


def _reader_for_context(context):
    from cloak.train.reward import _read_batch

    return lambda q: _read_batch([q], context)[0]


def _reader_mc_for_context(context):
    from cloak.train.reward import _read_batch, canon

    def read(q, options):
        prompt = q + "\nOptions:\n" + "\n".join(f"- {o}" for o in options)
        answer = _read_batch([prompt], context)[0]
        answer_c = canon(answer)
        for option in options:
            option_c = canon(option)
            if option_c and (option_c in answer_c or answer_c in option_c):
                return option
        return None
    return read


def build_ladder(args):
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs
    from cloak.train import ladder_probes as lp
    from cloak.train.roundtrip import roundtrip_batch

    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    flat_rung0 = _validated_rung0_lookup()
    ladder_out, decision_out = {}, {}
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"corpora": {}}
    report["th"] = args.th
    report.setdefault("corpora", {})

    for corpus in args.corpora.split(","):
        docs = load_task_docs(corpus, args.n_docs)
        per_doc = env["corpora"].get(corpus, {})
        rows = [d for d in docs if d["id"] in per_doc and per_doc[d["id"]]["spans"]]
        spans_of = {
            d["id"]: [s for s in per_doc[d["id"]]["spans"] if _span_rungs(s)]
            for d in rows
        }

        jobs, meta = [], []
        for d in rows:
            spans = per_doc[d["id"]]["spans"]
            ph_choice = {s["surface"].lower():
                         s["actions"][next(i for i, a in enumerate(s["actions"])
                                           if a["mode"] == "placeholder")]
                         for s in spans}
            lo_doc, lo_R = assemble(d["text"], art[corpus][d["id"]]["tau_walk"][1],
                                    spans, ph_choice)
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=args.workers)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = r["out_final"]
        out_hi_of = {doc_id: pair["hi"] for doc_id, pair in anchor.items() if "hi" in pair}

        ladders = lp.ladder_probes_for_docs(rows, spans_of, corpus, workers=args.workers)
        decisions = lp.decision_probes_for_docs(rows, out_hi_of, corpus, workers=args.workers)
        stats = {"docs": 0, "spans": 0, "rung_candidates": 0, "rung_kept": 0,
                 "decisions_kept": 0}
        for d in rows:
            doc_id = d["id"]
            if doc_id not in anchor:
                continue
            entries = _with_validated_rung0(
                ladders.get(doc_id, []), spans_of.get(doc_id, []), flat_rung0.get(doc_id, {})
            )
            kept, ladder_rows = lp.validate_ladder(
                entries,
                _reader_for_context(anchor[doc_id]["hi"]),
                _reader_for_context(anchor[doc_id]["lo"]),
                args.th,
            )
            ladder_out[doc_id] = _validated_entries(entries, ladder_rows)

            decision_entries = [
                {**e, "detected_spans": spans_of.get(doc_id, [])}
                for e in decisions.get(doc_id, [])
            ]
            kept_decisions, decision_rows = lp.validate_decisions(
                decision_entries,
                _reader_mc_for_context(anchor[doc_id]["hi"]),
                _reader_mc_for_context(anchor[doc_id]["lo"]),
            )
            decision_out[doc_id] = _validated_entries(
                [{k: v for k, v in e.items() if k != "detected_spans"}
                 for e in decision_entries],
                decision_rows,
            )

            stats["docs"] += 1
            stats["spans"] += len(spans_of.get(doc_id, []))
            stats["rung_candidates"] += len(entries)
            stats["rung_kept"] += len(kept)
            stats["decisions_kept"] += len(kept_decisions)

        row = ladder_health_row(**stats)
        report["corpora"].setdefault(corpus, {}).update(row)
        print(f"[{corpus} ladder] {row}", flush=True)

    lp.LADDER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    lp.LADDER_CACHE.write_text(json.dumps(ladder_out, indent=1))
    lp.DECISION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    lp.DECISION_CACHE.write_text(json.dumps(decision_out, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {lp.LADDER_CACHE} + {lp.DECISION_CACHE} + {REPORT}")


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs, refs_of
    from cloak.train.probes import PROMPT_VERSION, TEACHER_MODEL, probes_for_docs
    from cloak.train.reward import canon, fact_f1s
    from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL, roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="clinical,enron,aeslc")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--th", type=float, default=TH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default="data/ranker_env.json",
                    help="ranker environment artifact (default: frozen env; pilot env to retarget)")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json",
                    help="arms artifact (default: frozen historical; must match --env)")
    ap.add_argument("--ladder", action="store_true",
                    help="build ladder and decision probes from cached anchors")
    args = ap.parse_args()

    if args.ladder:
        build_ladder(args)
        return

    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = prev.get("docs", {}) if isinstance(prev, dict) else {}
    report = {"th": args.th, "corpora": {}}

    for corpus in args.corpora.split(","):
        docs = load_task_docs(corpus, args.n_docs)
        per_doc = env["corpora"].get(corpus, {})
        rows = [d for d in docs if d["id"] in per_doc and per_doc[d["id"]]["spans"]]
        # 1. candidate probes (teacher, cached; R = artifact tau_walk R)
        R_of = {d["id"]: art[corpus][d["id"]]["tau_walk"][1] for d in rows}
        cands = probes_for_docs(rows, R_of, workers=args.workers)
        # 2. anchor round trips: ceiling (doc_orig, R=[]) + floor (all-placeholder)
        jobs, meta = [], []
        for d in rows:
            spans = per_doc[d["id"]]["spans"]
            ph_choice = {s["surface"].lower():
                         s["actions"][next(i for i, a in enumerate(s["actions"])
                                           if a["mode"] == "placeholder")]
                         for s in spans}
            lo_doc, lo_R = assemble(d["text"], art[corpus][d["id"]]["tau_walk"][1],
                                    spans, ph_choice)
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                jobs.append({"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []})
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=args.workers)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = r["out_final"]
        # 3. validate (per QUESTION) + split/floor (per FACT)
        stats = {"docs": 0, "kept_facts": [], "kept_questions": [], "rej_c": 0, "rej_f": 0,
                 "cand": 0, "excluded_docs": [], "hi_kept": []}
        for d in rows:
            ps = cands.get(d["id"], [])
            if not ps or d["id"] not in anchor:
                # span-bearing doc with no candidate probes (or no anchor) is excluded, not
                # silently dropped — it contributes no RL reward signal
                stats["excluded_docs"].append(d["id"])
                continue
            hi = fact_f1s(anchor[d["id"]]["hi"], ps)
            lo = fact_f1s(anchor[d["id"]]["lo"], ps)
            kept, rc, rf = validate_probes(ps, hi, lo, args.th)
            hi_kept = [h for _p, h, l in zip(ps, hi, lo) if h >= args.th and l < args.th]
            train_q, heldout_q, n_train_facts = split_by_fact(kept, args.seed)
            out[d["id"]] = {"train": train_q, "heldout": heldout_q,
                            "rejected": {"ceiling": rc, "floor": rf}}
            stats["docs"] += 1
            stats["cand"] += len(ps)
            stats["kept_questions"].append(len(kept))
            stats["kept_facts"].append(len({canon(p["surface"]) for p in kept}))
            stats["hi_kept"].extend(hi_kept)
            stats["rej_c"] += len(rc)
            stats["rej_f"] += len(rf)
            # exclusion floor: < 3 DISTINCT FACTS in the train split (not questions)
            if n_train_facts < 3:
                stats["excluded_docs"].append(d["id"])
        n = max(stats["docs"], 1)
        report["corpora"][corpus] = {
            "docs": stats["docs"],
            "kept_facts_mean": round(sum(stats["kept_facts"]) / n, 2),
            "kept_questions_mean": round(sum(stats["kept_questions"]) / n, 2),
            "kept_min": min(stats["kept_facts"], default=0),
            "ceiling_reject_rate": round(stats["rej_c"] / max(stats["cand"], 1), 3),
            "floor_reject_rate": round(stats["rej_f"] / max(stats["cand"], 1), 3),
            "reader_hi_f1_kept_mean": (round(sum(stats["hi_kept"]) / len(stats["hi_kept"]), 3)
                                       if stats["hi_kept"] else None),
            "excluded_docs": stats["excluded_docs"]}
        print(f"[{corpus}] {report['corpora'][corpus]}", flush=True)

    from cloak.train.reward import QA_MODEL
    artifact = {"meta": {"rt_model": RT_MODEL, "rt_base_url": RT_BASE_URL,
                         "teacher": TEACHER_MODEL, "reader": QA_MODEL, "scorer": "fact_score_v2",
                         "th": args.th, "pv": PROMPT_VERSION, "env_path": args.env,
                         "built_at": datetime.datetime.now().isoformat(timespec="seconds")},
                "docs": out}
    OUT.write_text(json.dumps(artifact, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT} + {REPORT}")


if __name__ == "__main__":
    main()
