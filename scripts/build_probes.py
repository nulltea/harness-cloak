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

from cloak.profile_match import match_spans_batch, span_key

TH = 0.5
OUT = Path("data/probes_validated.json")
LADDER_VALIDATED_OUT = Path("data/probes_ladder_validated.json")
GEN_REJECTS_OUT = Path("results/ladder_gen_rejects.json")
GENERATIONS_OUT = Path("results/ladder_generations.json")
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


def validated_artifact(ladder_out, decision_out, meta):
    from cloak.train import ladder_probes as lp
    from cloak.train.reward import QA_MODEL
    from cloak.train.roundtrip import RT_MODEL

    return {
        "meta": {
            "teacher": meta.get("teacher", lp.TEACHER_MODEL),
            "reader": meta.get("reader", QA_MODEL),
            "rt_model": meta.get("rt_model", RT_MODEL),
            "th": meta["th"],
            "ladder_pv": meta.get("ladder_pv", lp.LADDER_PV),
            "decision_pv": meta.get("decision_pv", lp.DECISION_PV),
            "corpora": meta["corpora"],
            "determinism": meta.get("determinism", "workers1"),
            "env_path": meta["env_path"],
            "built_at": meta["built_at"],
        },
        "ladder": ladder_out,
        "decisions": decision_out,
    }


def _span_rungs(span):
    from cloak.train.ladder_probes import rung_phrases, span_levels

    levels = span_levels(span)
    return rung_phrases(span["surface"], levels) if levels else []


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


def _with_validated_rung0(entries, spans, validated, teacher=None, pv=None):
    from cloak.train.reward import canon

    by_surface = {canon(s["surface"]): _span_rungs(s) for s in spans if _span_rungs(s)}

    def validated_row(surface, q, rungs):
        row = {
            "surface": surface,
            "rung": 0,
            "q": q,
            "a": surface,
            "rungs": rungs,
            "source": "probes_validated",
        }
        if teacher is not None:
            row["teacher"] = teacher
        if pv is not None:
            row["pv"] = pv
        return row

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
                out.append(validated_row(row["surface"],
                                         probe.get("question") or probe.get("q"),
                                         by_surface[key]))
                replaced.add(key)
            continue
        out.append(row)
    present_r0 = {(canon(e.get("surface", "")), e.get("rung")) for e in out}
    for s in spans:
        key = canon(s["surface"])
        rungs = by_surface.get(key)
        if rungs and key in validated and (key, 0) not in present_r0:
            probe = validated[key]
            out.append(validated_row(s["surface"], probe.get("question") or probe.get("q"),
                                     rungs))
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
    from cloak.train.reward import _mc_pick, _read_mc_batch, decision_prompt

    def read(q, options):
        return _mc_pick(_read_mc_batch([decision_prompt(q, options)], context)[0], options)
    return read


def build_ladder(args):
    from build_arms_artifact import load_artifact
    from train_ranker import assemble

    from cloak.corpora import load_task_docs
    from cloak.tasks import SCHEMA_CORPORA
    from cloak.train import ladder_probes as lp
    from cloak.train.roundtrip import roundtrip_batch

    teacher_model = args.teacher_model or lp.TEACHER_MODEL
    teacher_base_url = args.teacher_base_url or lp.LOCAL_BASE_URL

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
                job = {"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []}
                if corpus in SCHEMA_CORPORA:
                    job["template"] = "schema"
                jobs.append(job)
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=1)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = {
                "out_p": r["out_p"],
                "out_final": r["out_final"],
            }
        out_hi_of = {
            doc_id: pair["hi"]["out_final"]
            for doc_id, pair in anchor.items()
            if "hi" in pair
        }

        ladders = lp.ladder_probes_for_docs(rows, spans_of, corpus, workers=args.workers,
                                            model=teacher_model, base_url=teacher_base_url)
        decisions = lp.decision_probes_for_docs(rows, out_hi_of, corpus, workers=args.workers,
                                                model=teacher_model, base_url=teacher_base_url)
        stats = {"docs": 0, "spans": 0, "rung_candidates": 0, "rung_kept": 0,
                 "decisions_kept": 0}
        for d in rows:
            doc_id = d["id"]
            if doc_id not in anchor:
                continue
            entries = _with_validated_rung0(
                ladders.get(doc_id, []),
                spans_of.get(doc_id, []),
                flat_rung0.get(doc_id, {}),
                teacher=teacher_model,
                pv=lp.LADDER_PV,
            )
            kept, ladder_rows = lp.validate_ladder(
                entries,
                _reader_for_context(anchor[doc_id]["hi"]["out_final"]),
                _reader_for_context(anchor[doc_id]["hi"]["out_p"]),
                _reader_for_context(anchor[doc_id]["lo"]["out_final"]),
                _reader_for_context(anchor[doc_id]["lo"]["out_p"]),
                args.th,
            )
            ladder_out[doc_id] = _validated_entries(entries, ladder_rows)

            decision_entries = [
                {**e, "detected_spans": spans_of.get(doc_id, [])}
                for e in decisions.get(doc_id, [])
            ]
            kept_decisions, decision_rows = lp.validate_decisions(
                decision_entries,
                _reader_mc_for_context(anchor[doc_id]["hi"]["out_p"]),
                _reader_mc_for_context(anchor[doc_id]["lo"]["out_p"]),
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

    artifact = validated_artifact(
        ladder_out,
        decision_out,
        {
            "th": args.th,
            "teacher": teacher_model,
            "teacher_base_url": teacher_base_url,
            "corpora": args.corpora.split(","),
            "env_path": args.env,
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    LADDER_VALIDATED_OUT.parent.mkdir(parents=True, exist_ok=True)
    LADDER_VALIDATED_OUT.write_text(json.dumps(artifact, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {LADDER_VALIDATED_OUT} + {REPORT}")


# Zero-shot GLiNER label (native knowledgator/gliner-pii-large-v1.0 training label) ->
# (runtime_type, role). Only `lattice` spans generate ladder probes, with levels sourced from
# lattice_profiles.json (the single source of truth). `placeholder`/`quasi` spans are detected
# so the floor anchor hides them (faithful all-sensitive-hidden baseline) but carry no lattice
# and produce no probes. Labels are the model's trained strings (model card), not free phrasing.
DETECT_LABELS = {
    # lattice-bearing (ladder probes; levels from lattice_profiles.json)
    "condition": ("health-condition", "lattice"),
    "drug": ("drug", "lattice"),
    "medical process": ("medical-procedure", "lattice"),
    "location city": ("LOC", "lattice"),
    "location country": ("LOC", "lattice"),
    "location state": ("LOC", "lattice"),
    # direct identifiers / no useful lattice -> placeholder (hidden in floor, no probes)
    "name": ("PERSON", "placeholder"),
    "first name": ("PERSON", "placeholder"),
    "last name": ("PERSON", "placeholder"),
    "age": ("age", "placeholder"),
    "gender": ("gender", "placeholder"),
    "marital status": ("marital-status", "placeholder"),
    "organization medical facility": ("organization-medical-facility", "placeholder"),
    "healthcare number": ("CODE", "placeholder"),
    "medical code": ("CODE", "placeholder"),
    # quasi -> rule-based range at substitution time; for the QA build it is hidden in the floor
    "dose": ("QUANTITY", "quasi"),
}


import re as _re

_DENIAL_CUES = _re.compile(r"\b(denies|denied|negative for|ruled out|no history of|"
                           r"without any|denying)\b", _re.IGNORECASE)


def _negated_or_screening(sent: str) -> bool:
    """A lattice fact mentioned only to be ruled out or screened for is not a documented finding
    and must not become a probe target (measured: mts/0 'congestion' lives only in 'Have you had
    any ... congestion? / No', which the teacher then fabricated a premise around). Conservative:
    fires on interrogative (screening-question) sentences and explicit denial cues ONLY — never on
    bare 'no/not', so 'no changes regarding hypertension' (patient HAS hypertension) is kept."""
    s = (sent or "").strip()
    return s.endswith("?") or bool(_DENIAL_CUES.search(s))


def _detect_docs(docs, model, threshold, max_words=320):
    """Fresh zero-shot detection -> {doc_id: [{surface, type, role, sent[, entry]}]}.
    `lattice` spans resolve through the shared retrieve-then-verify matcher
    (cloak.profile_match.match_spans_batch - same machinery as the substitutor) and are
    deduped per (runtime_type, matched canonical entry), so co-referent surfaces collapse to
    one span per document; matcher abstain drops the span (as a no-profile span is dropped
    today). `placeholder`/`quasi` spans are deduped per (surface, type). GPU (GLiNER)."""
    import torch
    from gliner import GLiNER

    from cloak.train.ladder_probes import sentence_of

    g = GLiNER.from_pretrained(model)
    if torch.cuda.is_available():
        g = g.to("cuda")
    labels = list(DETECT_LABELS)
    out = {}
    for d in docs:
        words = d["text"].split()
        seen, cands = set(), []
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            for e in g.predict_entities(piece, labels, threshold=threshold):
                surface, (rtype, role) = e["text"].strip(), DETECT_LABELS[e["label"]]
                key = (surface.lower(), rtype)
                if not surface or key in seen:
                    continue
                seen.add(key)
                sent = sentence_of(d["text"], surface)
                # a lattice fact mentioned only under negation/screening is not a probe target
                if role == "lattice" and _negated_or_screening(sent):
                    continue
                cands.append({"surface": surface, "type": rtype, "role": role, "sent": sent})
        lattice_cands = [c for c in cands if c["role"] == "lattice"]
        matches = match_spans_batch(
            [(c["surface"], c["type"], c["sent"]) for c in lattice_cands])
        spans, seen_entries = [], set()
        for c in cands:
            if c["role"] != "lattice":
                spans.append(c)
                continue
            m = matches.get(span_key(c["surface"], c["type"]))
            if m is None:
                continue  # abstain: not a probe span, drop (current no-profile behavior)
            entry_key = (c["type"], m.entry)
            if entry_key in seen_entries:
                continue  # co-referent duplicate within this document
            seen_entries.add(entry_key)
            spans.append({**c, "entry": m.entry})
        out[d["id"]] = spans
    return out


def _all_placeholder(text, spans):
    """Floor doc_p + R: replace EVERY detected sensitive span's surface (lattice + placeholder +
    quasi) with a typed placeholder — the faithful all-sensitive-hidden floor anchor."""
    from cloak.train.reward import generalize_text

    R, counts = [], {}
    for s in spans:
        t = s["type"]
        counts[t] = counts.get(t, 0) + 1
        ph = f"<{t.upper().replace('-', '_')}_{counts[t]}>"
        R.append({"surface": s["surface"], "type": t, "action": "placeholder", "replacement": ph})
    return generalize_text(text, R), R


def build_ladder_detected(args):
    """Ladder/decision QA build with FRESH detection + levels from lattice_profiles.json (no env).

    Spans: zero-shot detection (args.detector_model). Levels: lattice_profiles.json (span_levels).
    Anchors: ceiling = Remote(task(doc_orig), R=[]); floor = Remote(task(all-placeholder)); schema
    prompt on SCHEMA_CORPORA. Determinism: anchors at workers=1.
    """
    from cloak.corpora import load_task_docs
    from cloak.tasks import SCHEMA_CORPORA
    from cloak.train import ladder_probes as lp
    from cloak.train.roundtrip import roundtrip_batch

    teacher_model = args.teacher_model or lp.TEACHER_MODEL
    teacher_base_url = args.teacher_base_url or lp.LOCAL_BASE_URL
    tag = args.out_tag or ""
    suffix = f".{tag}" if tag else ""
    validated_out = LADDER_VALIDATED_OUT.with_suffix(f"{suffix}.json")
    rejects_out = GEN_REJECTS_OUT.with_suffix(f"{suffix}.json")
    generations_out = GENERATIONS_OUT.with_suffix(f"{suffix}.json")
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"corpora": {}}
    report["th"] = args.th
    report.setdefault("corpora", {})
    ladder_out, decision_out = {}, {}
    all_gen_rejects = {}
    generations = {}   # doc_id -> {out_hi, out_lo, ladder_raw, decision_raw} for later analysis

    for corpus in args.corpora.split(","):
        all_docs = load_task_docs(corpus, args.n_docs)
        detected = _detect_docs(all_docs, args.detector_model, args.detector_threshold)
        # lattice spans feed the ladder; ALL detected spans feed the floor anonymization
        lattice_of = {doc_id: [s for s in spans if s["role"] == "lattice"]
                      for doc_id, spans in detected.items()}
        rows = [d for d in all_docs if lattice_of.get(d["id"])]
        spans_of = lattice_of  # ladder_probes_for_docs / _with_validated_rung0 read lattice spans
        print(f"[{corpus}] {len(rows)}/{len(all_docs)} docs with lattice spans; "
              f"{sum(len(lattice_of[d['id']]) for d in rows)} lattice + "
              f"{sum(len(detected[d['id']]) for d in rows)} total detected", flush=True)

        jobs, meta = [], []
        for d in rows:
            lo_doc, lo_R = _all_placeholder(d["text"], detected[d["id"]])
            for kind, doc_p, R in (("hi", d["text"], []), ("lo", lo_doc, lo_R)):
                job = {"corpus": corpus, "doc_p": doc_p, "R": R, "probes": []}
                if corpus in SCHEMA_CORPORA:
                    job["template"] = "schema"
                jobs.append(job)
                meta.append((d["id"], kind))
        outs = roundtrip_batch(jobs, workers=1)
        anchor = {}
        for (doc_id, kind), r in zip(meta, outs):
            anchor.setdefault(doc_id, {})[kind] = {"out_p": r["out_p"], "out_final": r["out_final"]}
        out_hi_of = {doc_id: pair["hi"]["out_final"]
                     for doc_id, pair in anchor.items() if "hi" in pair}

        all_surfaces_of = {doc_id: [s["surface"] for s in spans]
                           for doc_id, spans in detected.items()}
        gen_rejects, ladder_raw, decision_raw = [], [], []
        ladders = lp.ladder_probes_for_docs(rows, spans_of, corpus, workers=args.workers,
                                            model=teacher_model, base_url=teacher_base_url,
                                            all_surfaces_of=all_surfaces_of,
                                            reject_sink=gen_rejects, gen_sink=ladder_raw)
        decisions = lp.decision_probes_for_docs(rows, out_hi_of, corpus, workers=args.workers,
                                                model=teacher_model, base_url=teacher_base_url,
                                                gen_sink=decision_raw)
        # record every generation (raw teacher replies + the anchors they were graded against)
        for d in rows:
            if d["id"] in anchor:
                generations[d["id"]] = {
                    "corpus": corpus,
                    "out_hi": anchor[d["id"]]["hi"]["out_final"],
                    "out_lo_p": anchor[d["id"]]["lo"]["out_p"],
                    "out_lo_final": anchor[d["id"]]["lo"]["out_final"],
                    "ladder_raw": [g for g in ladder_raw if g["doc_id"] == d["id"]],
                    "decision_raw": [g for g in decision_raw if g["doc_id"] == d["id"]],
                }
        stats = {"docs": 0, "spans": 0, "rung_candidates": 0, "rung_kept": 0, "decisions_kept": 0}
        for d in rows:
            doc_id = d["id"]
            if doc_id not in anchor:
                continue
            entries = _with_validated_rung0(ladders.get(doc_id, []), spans_of.get(doc_id, []), {},
                                            teacher=teacher_model, pv=lp.LADDER_PV)
            kept, ladder_rows = lp.validate_ladder(
                entries,
                _reader_for_context(anchor[doc_id]["hi"]["out_final"]),
                _reader_for_context(anchor[doc_id]["hi"]["out_p"]),
                _reader_for_context(anchor[doc_id]["lo"]["out_final"]),
                _reader_for_context(anchor[doc_id]["lo"]["out_p"]),
                args.th,
            )
            ladder_out[doc_id] = _validated_entries(entries, ladder_rows)

            decision_entries = [{**e, "detected_spans": spans_of.get(doc_id, [])}
                                for e in decisions.get(doc_id, [])]
            kept_decisions, decision_rows = lp.validate_decisions(
                decision_entries,
                _reader_mc_for_context(anchor[doc_id]["hi"]["out_p"]),
                _reader_mc_for_context(anchor[doc_id]["lo"]["out_p"]),
            )
            decision_out[doc_id] = _validated_entries(
                [{k: v for k, v in e.items() if k != "detected_spans"} for e in decision_entries],
                decision_rows,
            )

            stats["docs"] += 1
            stats["spans"] += len(spans_of.get(doc_id, []))
            stats["rung_candidates"] += len(entries)
            stats["rung_kept"] += len(kept)
            stats["decisions_kept"] += len(kept_decisions)

        row = ladder_health_row(**stats)
        report["corpora"].setdefault(corpus, {}).update(row)
        all_gen_rejects[corpus] = gen_rejects
        import collections as _c
        print(f"[{corpus} ladder/detect] {row}  gen_rejects="
              f"{dict(_c.Counter(r['gate'] for r in gen_rejects))}", flush=True)

    artifact = validated_artifact(ladder_out, decision_out, {
        "th": args.th, "teacher": teacher_model, "teacher_base_url": teacher_base_url,
        "corpora": args.corpora.split(","), "spans_source": f"detected:{args.detector_model}",
        "env_path": None, "built_at": datetime.datetime.now().isoformat(timespec="seconds")})
    validated_out.parent.mkdir(parents=True, exist_ok=True)
    validated_out.write_text(json.dumps(artifact, indent=1))
    rejects_out.parent.mkdir(parents=True, exist_ok=True)
    rejects_out.write_text(json.dumps(all_gen_rejects, indent=1))
    generations_out.write_text(json.dumps(
        {"meta": artifact["meta"], "docs": generations}, indent=1))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {validated_out} + {rejects_out} + {generations_out} + {REPORT}", flush=True)


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
                    help="build ladder and decision probes from cached anchors (env spans_source)")
    ap.add_argument("--detect", action="store_true",
                    help="ladder/decision build with FRESH detection + levels from "
                         "lattice_profiles.json (no env). Overrides --ladder.")
    ap.add_argument("--detector-model", default="knowledgator/gliner-pii-large-v1.0",
                    help="zero-shot GLiNER detector for --detect")
    ap.add_argument("--detector-threshold", type=float, default=0.35)
    ap.add_argument("--out-tag", default="",
                    help="suffix for --detect outputs (probes_ladder_validated.<tag>.json etc.), "
                         "so a teacher-model sweep writes side-by-side artifacts")
    ap.add_argument("--teacher-model", default="",
                    help="override the ladder/decision teacher (default: ladder_probes.TEACHER_MODEL). "
                         "Remote OpenRouter model ids (e.g. nvidia/nemotron-3-super-120b-a12b:free) need "
                         "--teacher-base-url https://openrouter.ai/api/v1 and OPENROUTER_API_KEY set.")
    ap.add_argument("--teacher-base-url", default="",
                    help="teacher base_url (default: local proxy; openrouter.ai for hosted teachers)")
    args = ap.parse_args()

    if args.detect:
        build_ladder_detected(args)
        return
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
