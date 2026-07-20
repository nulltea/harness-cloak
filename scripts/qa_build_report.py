#!/usr/bin/env python3
"""Generate a Markdown diagnostics report from a QA-builder-v2 utility artifact.

Reads a finished build directory (the `*.utility.json` plus its `*.qa-audit.jsonl`
sibling) and emits:
  * `<stem>.report.md`                     -- the human-readable report
  * `<stem>.bad_lattice_profiles.json`     -- the lattice-triage detail (Tier 1/2)

It re-derives everything from the artifacts (no reader/teacher re-run, zero cost), so
it is safe to run after any build. Relation-pipeline view only (delivered/coverage
span assertions are summarised in one line, not itemised).

Usage:
    python scripts/qa_build_report.py results/qa_v2_aci_full
    python scripts/qa_build_report.py results/qa_v2_aci_full --out /tmp/report.md
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def _find_artifact(results_dir: Path) -> Path:
    # the main artifact ends exactly in `.utility.json`; siblings are `.utility.qa-audit.json`, etc.
    cands = [p for p in results_dir.glob("*.utility.json") if p.name.endswith(".utility.json")
             and not any(seg in p.name for seg in (".qa-audit.", ".assertions.", ".qa-pairs."))]
    if len(cands) != 1:
        sys.exit(f"expected exactly one *.utility.json in {results_dir}, found {[p.name for p in cands]}")
    return cands[0]


def _occurrence_surface_map(documents: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in documents.values():
        for occ in rec.get("occurrences") or []:
            out[str(occ.get("occurrence_id"))] = occ.get("surface")
    return out


def _doc_short(doc_id: str) -> str:
    return doc_id.split("/")[-1]


def _iter_audit(audit_path: Path):
    with audit_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _lattice_triage(audit_path: Path, occ_surface: dict[str, str], rel_gen: dict) -> dict:
    """Split three_point_gate_failed by reader score pattern and collect the locator-probe hits.

    orig!=rep means the reader proved capability on one side (so it is NOT a reader-can't-answer
    failure): (1,0,0) representative render unreadable; (0,1,0) specific surface not credited at the
    representative level. The build's `lattice_probe` fires on (1,0,0) when a COARSER locator level
    reads. IMPORTANT: this probe is FALSE-POSITIVE-prone -- a vaguer locator is simply easier for a
    weak reader, and the true cause is usually answer ambiguity. Callers must cross-check each hit
    against kept relations (a level used by a kept QA-pair is readable, so the flag is spurious).
    """
    pattern = collections.Counter()
    probes: dict[tuple, dict] = {}
    pool = collections.defaultdict(lambda: {"over_coarse": 0, "inverted": 0, "docs": set()})
    n_cases = 0
    for rec in _iter_audit(audit_path):
        if rec.get("code") != "relation_rejected:three_point_gate_failed":
            continue
        ev = (rec.get("evidence") or {}).get("evidence") or {}
        val = ev.get("validation") or {}
        scores = val.get("scores") or {}
        thr = (val.get("stability") or {}).get("threshold", 1.0)
        o = 1 if scores.get("original", 0.0) >= thr else 0
        rp = 1 if scores.get("representative", 0.0) >= thr else 0
        ph = 1 if scores.get("placeholder", 0.0) >= thr else 0
        pattern[(o, rp, ph)] += 1
        if o == rp:
            continue
        n_cases += 1
        doc = _doc_short(rec["doc_id"])
        probe = ev.get("lattice_probe")
        if probe:
            key = (probe["runtime_type"], probe["surface"], probe["unreadable_level"])
            entry = probes.setdefault(key, {
                "runtime_type": probe["runtime_type"], "surface": probe["surface"],
                "chosen_level": probe["unreadable_level"],
                "readable_coarser_level": probe.get("readable_coarser_level"),
                "chain": probe.get("chain"), "docs": set(),
            })
            entry["docs"].add(doc)
        for arg in ev.get("arguments") or []:
            surface = occ_surface.get(str(arg.get("occurrence_id")))
            level = arg.get("support_property") or arg.get("literal")
            rtype = arg.get("runtime_type")
            if not surface or not level or surface == level:
                continue  # only actually-generalised args can be a lattice culprit
            pe = pool[(rtype, surface, level)]
            pe["over_coarse" if o == 1 else "inverted"] += 1
            pe["docs"].add(doc)

    tier1 = [{k: v for k, v in e.items() if k != "docs"} | {"docs": sorted(e["docs"])}
             for e in probes.values()]
    probe_keys = set(probes)
    tier2 = [{
        "runtime_type": rt, "surface": surf, "chosen_level": lvl,
        "over_coarse": e["over_coarse"], "inverted": e["inverted"],
        "probe_confirmed": (rt, surf, lvl) in probe_keys, "docs": sorted(e["docs"]),
    } for (rt, surf, lvl), e in sorted(pool.items(), key=lambda kv: -(kv[1]["over_coarse"] + kv[1]["inverted"]))]
    return {"pattern": pattern, "n_cases": n_cases, "tier1": tier1, "tier2": tier2}


_CLINICAL_TYPES = {"health-condition", "medical-procedure", "drug"}
_AXIS_LABEL = re.compile(
    r"administration, physiological systems"
    r"|, (?:bypass|replacement|revision|repair|excision|introduction|unspecified)", re.I)


def _level_defect(node: str) -> str | None:
    """Objective data-quality defect in a generalization LEVEL string (independent of the reader).

    A malformed level harms QA quality even when the reader happens to match it, so this is a
    separate axis from reader-readability. Flags grouping-node artifacts, SNOMED axis descriptors
    (hierarchy labels, not usable concepts), 'X administration' boilerplate, and verbose FSNs.
    """
    toks = node.split()
    if "family" in toks:
        return "grouping-node-artifact"
    if _AXIS_LABEL.search(node):
        return "snomed-axis-label"
    if node.endswith("administration") or "administration," in node:
        return "admin-boilerplate"
    if ("," in node or "(" in node) and len(toks) >= 4:
        return "fsn-verbose"
    return None


def _surface_garbled(surface: str) -> bool:
    # a duplicated >=4-char run in the profile's surface key (detector/normalisation artifact),
    # e.g. "echocardiogram echocardiocardiogram", "ultrasound elasto elastography".
    return bool(re.search(r"(\w{4,})\1", surface.replace(" ", "")))


def _lattice_quality_audit(documents: dict, used: dict, used_kept: dict) -> list[dict]:
    """Audit every clinical decision's level chain for malformed data. Returns rows sorted by
    defect severity then usage. `used`/`used_kept` count how often a level was selected as a
    relation argument (a bad level actually chosen is higher priority than one idling in the ladder)."""
    severity = {"grouping-node-artifact": 0, "snomed-axis-label": 1, "admin-boilerplate": 2,
                "fsn-verbose": 3, "garbled-surface-key": 4}
    rows, seen = [], set()
    for rec in documents.values():
        for dec in rec.get("decisions") or []:
            rt = dec.get("runtime_type")
            if rt not in _CLINICAL_TYPES:
                continue
            surface = dec.get("canonical_key") or ""
            nodes = [n["node"] for n in dec.get("semantic_chain") or []
                     if n["node"] not in ("keep", "placeholder")]
            key = (rt, surface, tuple(nodes))
            if key in seen:
                continue
            seen.add(key)
            defects = {n: d for n in nodes if (d := _level_defect(n))}
            garbled = _surface_garbled(surface)
            if not defects and not garbled:
                continue
            tags = sorted(set(defects.values()) | ({"garbled-surface-key"} if garbled else set()),
                          key=lambda t: severity.get(t, 9))
            rows.append({
                "runtime_type": rt, "surface": surface, "chain": nodes,
                "defects": defects, "garbled_surface": garbled, "tags": tags,
                "max_used": max((used.get(n, 0) for n in defects), default=0),
                "max_kept": max((used_kept.get(n, 0) for n in defects), default=0),
            })
    rows.sort(key=lambda r: (severity.get(r["tags"][0], 9), -r["max_used"], r["surface"]))
    return rows


def _gate_scores(audit_path: Path) -> dict[tuple, tuple]:
    """(doc, subject_level, object_level) -> binarised (orig, rep, ph) reader triple."""
    out: dict[tuple, tuple] = {}
    for rec in _iter_audit(audit_path):
        if rec.get("code") != "relation_rejected:three_point_gate_failed":
            continue
        ev = (rec.get("evidence") or {}).get("evidence") or {}
        val = ev.get("validation") or {}
        sc = val.get("scores") or {}
        thr = (val.get("stability") or {}).get("threshold", 1.0)
        args = ev.get("arguments") or []
        subj = next((a.get("support_property") or a.get("literal")
                     for a in args if a.get("role") == "subject"), None)
        obj = next((a.get("support_property") or a.get("literal")
                    for a in args if a.get("role") == "object"), None)
        out[(_doc_short(rec["doc_id"]), subj, obj)] = (
            1 if sc.get("original", 0.0) >= thr else 0,
            1 if sc.get("representative", 0.0) >= thr else 0,
            1 if sc.get("placeholder", 0.0) >= thr else 0,
        )
    return out


def _classify_lost_gate_failures(rel_gen: dict, kept_triples: set, competing: dict,
                                 gate_scores: dict, bad_levels: dict) -> dict:
    """Classify GENUINELY-LOST three_point_gate_failed relations (fact triple never kept in the doc).

    Modes: ambiguity (subject has >=2 same-type answers -> no unique QA), bad-data (an argument level
    is a malformed lattice string), reader (real fact lost to the local reader: capability=orig
    readable/generalized not, or scoring/chain=generalized readable/original not credited), legit
    (unsupported -- reader fails on the original render too -- or self/tautology).
    """
    def lv(a):
        return a.get("support_property") or a.get("literal")

    detail = collections.Counter()
    examples = collections.defaultdict(list)
    total = lost = 0
    for doc, props in rel_gen.items():
        d = _doc_short(doc)
        for p in props:
            if p.get("reason") != "three_point_gate_failed":
                continue
            total += 1
            subj = next((lv(a) for a in p["arguments"] if a["role"] == "subject"), None)
            obj = next((lv(a) for a in p["arguments"] if a["role"] == "object"), None)
            if (doc, p["relation"], subj, obj) in kept_triples:
                continue  # recovered elsewhere -> not a genuine loss
            lost += 1
            pat = gate_scores.get((d, subj, obj))
            bad = next((bad_levels[x] for x in (subj, obj) if x in bad_levels), None)
            if subj is not None and subj == obj:
                cat = "legit:self-tautology"
            elif bad:
                cat = f"bad-data:{bad}"
            elif len(competing[(d, p["relation"], subj)]) > 1:
                cat = "ambiguity:multi-answer"
            elif pat == (1, 0, 0):
                cat = "reader:capability"
            elif pat == (0, 0, 0):
                cat = "legit:unsupported"
            elif pat == (0, 1, 0):
                cat = "reader:scoring-chain"
            else:
                cat = "unknown:no-score"
            detail[cat] += 1
            if len(examples[cat]) < 6:
                examples[cat].append(f"{d} {p['relation']}({subj} → {obj})")

    rollup = collections.Counter()
    for cat, n in detail.items():
        rollup[cat.split(":", 1)[0]] += n
    return {"total": total, "lost": lost, "recovered": total - lost,
            "detail": detail, "rollup": rollup, "examples": examples}


def _resolve_profile_entries(tier1: list[dict], profiles_path: Path) -> None:
    """Annotate each Tier-1 row with the canonical lattice_profiles.json key (surfaces are often aliases)."""
    if not profiles_path.exists():
        return
    try:  # tolerate a mid-edit / malformed profiles file -- the annotation is best-effort
        profiles = json.loads(profiles_path.read_text()).get("profiles", {})
    except (json.JSONDecodeError, OSError):
        return

    def canon_of(rtype: str, surface: str):
        for canon, ent in profiles.get(rtype, {}).items():
            aliases = {a.lower() for a in (ent.get("aliases") or [])}
            if surface.lower() == canon.lower() or surface.lower() in aliases:
                return canon
        return None

    for row in tier1:
        row["profile_entry"] = canon_of(row["runtime_type"], row["surface"])


def build_report(results_dir: Path, out_path: Path | None) -> Path:
    artifact_path = _find_artifact(results_dir)
    stem = artifact_path.name[: -len(".utility.json")]
    audit_path = results_dir / f"{stem}.utility.qa-audit.jsonl"
    if not audit_path.exists():
        sys.exit(f"missing audit sibling: {audit_path}")

    art = json.loads(artifact_path.read_text())
    occ_surface = _occurrence_surface_map(art["documents"])
    rel_gen = art.get("relation_generation", {})
    gleaning = art.get("relation_gleaning", {})
    assertions = art.get("assertions", {})

    # --- kept relations (deduped final) ---
    kept_by_doc = collections.Counter()
    kept_by_type = collections.Counter()
    for a in assertions.values():
        if a.get("subtype") == "contextual_relation":
            kept_by_doc[_doc_short(a["doc_id"])] += 1
            kept_by_type[a.get("relation")] += 1
    total_kept = sum(kept_by_doc.values())

    # --- kept levels per doc + competing same-type answers per (doc, relation, subject) ---
    # A level used by a kept QA-pair is provably readable; a subject with >=2 same-type answers is
    # the real driver of most three_point_gate_failed (reader cannot uniquely pick the target).
    kept_levels_by_doc: dict[str, set] = collections.defaultdict(set)
    competing: dict[tuple, set] = collections.defaultdict(set)
    kept_triples: set = set()  # (doc, relation, subject_level, object_level) kept anywhere
    used_levels = collections.Counter()       # level -> times chosen as a relation arg
    used_levels_kept = collections.Counter()   # level -> times chosen in a KEPT relation
    for doc, props in rel_gen.items():
        d = _doc_short(doc)
        for p in props:
            subj = next((a.get("support_property") for a in p["arguments"] if a["role"] == "subject"), None)
            obj = next((a.get("support_property") or a.get("literal")
                        for a in p["arguments"] if a["role"] == "object"), None)
            competing[(d, p["relation"], subj)].add(obj)
            if p.get("status") == "kept":
                kept_triples.add((doc, p["relation"], subj, obj))
            for a in p["arguments"]:
                lvl = a.get("support_property") or a.get("literal")
                if lvl:
                    used_levels[lvl] += 1
                    if p.get("status") == "kept":
                        used_levels_kept[lvl] += 1
            if p.get("status") == "kept":
                for a in p["arguments"]:
                    kept_levels_by_doc[d].add(a.get("support_property") or a.get("literal"))

    # --- proposals: approved (gate-passed) vs drop channels, per run ---
    approved = collections.defaultdict(lambda: [0, 0])  # doc -> [primary, gleaning]
    drop_channels = collections.Counter()
    run_status = collections.Counter()
    for doc, props in rel_gen.items():
        d = _doc_short(doc)
        for p in props:
            run = "primary" if p.get("run_id") == "primary" else "gleaning"
            if p.get("status") == "kept":
                approved[d][0 if run == "primary" else 1] += 1
                run_status[(run, "kept")] += 1
            else:
                drop_channels[p.get("reason")] += 1
                run_status[(run, "rejected")] += 1
    total_approved = sum(sum(v) for v in approved.values())

    # --- merge disposition + gleaning re-authoring of already-kept facts ---
    merge = collections.Counter()
    triggered = 0
    for g in gleaning.values():
        if g.get("triggered"):
            triggered += 1
        for k, v in (g.get("merge_disposition") or {}).items():
            merge[k] += v

    reauth = collections.Counter()  # status -> count of gleaning proposals re-authoring a primary-kept pair
    for doc, props in rel_gen.items():
        kept_primary = {(p["relation"], tuple(a.get("span_label") for a in p["arguments"]))
                        for p in props if p.get("run_id") == "primary" and p.get("status") == "kept"}
        for p in props:
            if p.get("run_id") == "primary":
                continue
            key = (p["relation"], tuple(a.get("span_label") for a in p["arguments"]))
            if key in kept_primary:
                reauth["kept" if p.get("status") == "kept" else "rejected"] += 1

    # --- context-literal + leakage sub-breakdowns ---
    def _lit_breakdown(reason: str):
        by_doc, examples = collections.Counter(), []
        for doc, props in rel_gen.items():
            for p in props:
                if p.get("reason") != reason:
                    continue
                by_doc[_doc_short(doc)] += 1
                lits = [a.get("literal") for a in p["arguments"] if a.get("kind") == "context" and a.get("literal")]
                examples.append((_doc_short(doc), p["relation"], lits, p.get("question")))
        return by_doc, examples

    ucl_by_doc, ucl_ex = _lit_breakdown("unknown_context_literal")
    leak_by_doc, leak_ex = _lit_breakdown("answer_leakage")

    # --- three_point_gate_failed cause classification (relation_generation view) ---
    gate_cause = collections.Counter()
    for doc, props in rel_gen.items():
        d = _doc_short(doc)
        for p in props:
            if p.get("reason") != "three_point_gate_failed":
                continue
            subj = next((a.get("support_property") for a in p["arguments"] if a["role"] == "subject"), None)
            gate_cause["answer_ambiguity" if len(competing[(d, p["relation"], subj)]) > 1
                        else "other"] += 1

    # --- lattice triage (probe hits; then cross-checked against kept relations) ---
    lattice = _lattice_triage(audit_path, occ_surface, rel_gen)
    _resolve_profile_entries(lattice["tier1"], Path("data/lattice_profiles/lattice_profiles.json"))
    for row in lattice["tier1"]:
        row["level_kept_in_doc"] = any(row["chosen_level"] in kept_levels_by_doc.get(d, set())
                                       for d in row["docs"])
    n_false = sum(1 for r in lattice["tier1"] if r["level_kept_in_doc"])

    # --- lattice data-quality audit (malformed level strings, independent of the reader) ---
    lattice_quality = _lattice_quality_audit(art["documents"], used_levels, used_levels_kept)
    bad_levels = {node: tag for row in lattice_quality for node, tag in row["defects"].items()}

    # --- classify genuinely-lost gate failures (fact never kept in the doc) ---
    lost = _classify_lost_gate_failures(rel_gen, kept_triples, competing,
                                        _gate_scores(audit_path), bad_levels)

    # consistency self-check (ponytail: the meaningful assertion this report rests on). Kept =
    # merge disposition (primary+gleaning) + deterministic reverse-framing relations (own run).
    merge_sum = merge.get("primary_only", 0) + merge.get("primary_preferred", 0) + merge.get("secondary_only", 0)
    reverse_kept = sum(1 for props in rel_gen.values() for p in props
                       if p.get("run_id") == "reverse_framing" and p.get("status") == "kept")
    consistency = ("OK" if merge_sum + reverse_kept == total_kept
                   else f"MISMATCH (merge {merge_sum} + reverse {reverse_kept} != kept {total_kept})")

    # ---------------- render markdown ----------------
    L: list[str] = []
    add = L.append
    add(f"# QA-builder-v2 report — `{results_dir}`\n")
    add(f"- Artifact: `{artifact_path.name}`  hash `{art.get('artifact_hash', '')[:19]}…`")
    add(f"- Documents: **{len(art['documents'])}**  ·  assertions: **{len(assertions)}**  ·  kept relations: **{total_kept}**")
    for pin in ("task_pin", "builder_pin"):
        if art.get(pin):
            add(f"- {pin}: `{art[pin]}`")
    for pin in ("reader_pin", "teacher_pin", "relation_teacher_pins"):
        if art.get(pin):
            add(f"- {pin}: `{json.dumps(art[pin])[:160]}`")
    add("")

    add("## Kept relations by type\n")
    add("| relation | kept |")
    add("|---|---|")
    for rel, n in kept_by_type.most_common():
        add(f"| {rel} | {n} |")
    add(f"| **total** | **{total_kept}** |\n")

    add("## Gate-approved vs kept\n")
    add(f"- **{total_approved}** proposals passed the three-point reader gate; **{total_kept}** kept.")
    add(f"- Gap = **{total_approved - total_kept}** = cross-run merge (`primary_preferred`).")
    add(f"- Merge disposition: `primary_only={merge.get('primary_only', 0)}`, "
        f"`primary_preferred={merge.get('primary_preferred', 0)}`, "
        f"`secondary_only={merge.get('secondary_only', 0)}`  ·  self-check: **{consistency}**")
    add(f"- Gleaning triggered on **{triggered}/{len(gleaning)}** docs.\n")

    add("### Per-doc approved (primary+gleaning) → kept\n")
    add("| doc | approved (P+G) | kept | merged |")
    add("|---|---|---|---|")
    for doc in sorted(art["documents"]):
        d = _doc_short(doc)
        p, g = approved.get(d, [0, 0])
        k = kept_by_doc.get(d, 0)
        add(f"| {d} | {p}+{g}={p + g} | {k} | {p + g - k} |")
    add("")

    add("## Relation drop channels\n")
    add("| reason | count |")
    add("|---|---|")
    for reason, n in drop_channels.most_common():
        add(f"| {reason} | {n} |")
    add("")

    add("## Gleaning redundancy (re-authoring already-kept facts)\n")
    total_glean = run_status[("gleaning", "kept")] + run_status[("gleaning", "rejected")]
    reauth_total = reauth["kept"] + reauth["rejected"]
    add(f"- **{reauth_total}/{total_glean}** gleaning proposals re-author a decision-pair primary already kept "
        f"(`ambiguous` target branch).")
    add(f"  - {reauth['kept']} came back kept → merged away as `primary_preferred`.")
    add(f"  - {reauth['rejected']} came back rejected → wasted paid attempts + inflated rejection channels.\n")

    add("## `unknown_context_literal`\n")
    add(f"- {sum(ucl_by_doc.values())} rejections across {len(ucl_by_doc)} docs: "
        f"{dict(ucl_by_doc)}")
    for doc, rel, lits, _q in ucl_ex[:20]:
        add(f"  - {doc} `{rel}` context-literal {lits}")
    add("")

    add("## `answer_leakage`\n")
    add(f"- {sum(leak_by_doc.values())} rejections across {len(leak_by_doc)} docs: {dict(leak_by_doc)}\n")

    add("## `three_point_gate_failed` — answerability failures\n")
    pat = lattice["pattern"]
    total_gate = sum(pat.values())
    non_lattice = sum(n for (o, r, _p), n in pat.items() if o == r)
    amb = gate_cause["answer_ambiguity"]
    add(f"- {total_gate} total. Dominant cause is **answer ambiguity**: **{amb}/{total_gate}** "
        f"({100 * amb // max(total_gate, 1)}%) have ≥2 competing same-type answers for the subject, "
        f"so the reader cannot uniquely produce the target in the generalised render.")
    add(f"- {lattice['n_cases']} show `orig≠rep`; {non_lattice} are `(0,0,0)`/other (reader can't "
        f"answer either render = unsupported/bad anchor).")
    add("- Score patterns `(orig,rep,ph)` [pass=`(1,1,0)`]:")
    for k in sorted(pat, key=lambda x: -pat[x]):
        tag = " ← orig≠rep" if k[0] != k[1] else ""
        add(f"  - `{k}`: {pat[k]}{tag}")
    add("")

    add("### `lattice_level_suspect` probe hits — CROSS-CHECKED (false-positive-prone)\n")
    add(f"The build's locator probe flagged these levels as \"unreadable\". **{n_false}/"
        f"{len(lattice['tier1'])} are FALSE positives**: the same level is used by a *kept* relation in "
        "the same doc, so it is provably readable — the real cause is answer ambiguity / reader "
        "precision, not lattice data. Do NOT edit `lattice_profiles.json` from this table.\n")
    add("| profile entry | type | flagged level | reads at (coarser) | docs | kept elsewhere? |")
    add("|---|---|---|---|---|---|")
    for row in sorted(lattice["tier1"], key=lambda r: (not r["level_kept_in_doc"], r["runtime_type"])):
        entry = row.get("profile_entry") or f"(alias/missing: {row['surface']})"
        flag = "**FALSE (kept)**" if row["level_kept_in_doc"] else "no kept sibling"
        add(f"| {entry} | {row['runtime_type']} | {row['chosen_level']} | "
            f"{row.get('readable_coarser_level') or ''} | {', '.join(row['docs'])} | {flag} |")
    add("")

    add("## Genuinely-lost relations — failure-mode classification\n")
    add(f"Of {lost['total']} `three_point_gate_failed`, **{lost['recovered']}** are recovered elsewhere "
        f"(same fact kept under another proposal) and **{lost['lost']}** are genuinely lost. "
        "\"Genuinely lost\" = the `(relation, subject_level, object_level)` fact is never kept in the doc.\n")
    add("| mode | count | share | meaning |")
    add("|---|---|---|---|")
    mode_meaning = {
        "ambiguity": "real fact, subject has ≥2 genuinely-distinct same-type answers → no unique free-form answer; second-call repair is attempted but cannot resolve it (a QA-contract limit, not reader/data)",
        "legit": "correctly rejected: unsupported (reader fails on the *original* render too) or self/tautology",
        "reader": "real fact lost to the local reader: capability (generalized term unreadable) or scoring/chain (specific answer not credited)",
        "bad-data": "an argument level is a malformed lattice string (family/axis/admin/fsn)",
        "unknown": "score-join missed (re-anchored surface)",
    }
    lost_n = max(lost["lost"], 1)
    for mode, n in lost["rollup"].most_common():
        add(f"| {mode} | {n} | {100 * n // lost_n}% | {mode_meaning.get(mode, '')} |")
    add("")
    add("Sub-modes:")
    for cat, n in lost["detail"].most_common():
        add(f"- `{cat}`: {n}  — e.g. {'; '.join(lost['examples'][cat][:3])}")
    add("")

    add("## Lattice data-quality audit (malformed level strings)\n")
    add("Direct audit of clinical generalization ladders in `lattice_profiles.json` — **independent of "
        "the reader**. A malformed level harms QA quality even when the reader matches it, so this is a "
        "separate axis from the answerability failures above. `used`/`kept` = times the level was chosen "
        "as a relation argument.\n")
    add(f"- **{len(lattice_quality)} clinical entities** carry ≥1 malformed level or a garbled surface key.\n")
    add("| type | profile entry (surface) | defect | offending level | used | kept |")
    add("|---|---|---|---|---|---|")
    for r in lattice_quality:
        if r["defects"]:
            for node, tag in r["defects"].items():
                add(f"| {r['runtime_type']} | {r['surface']} | {tag} | {node} | "
                    f"{used_levels.get(node, 0)} | {used_levels_kept.get(node, 0)} |")
        if r["garbled_surface"]:
            add(f"| {r['runtime_type']} | {r['surface']} | garbled-surface-key | (surface key itself) | — | — |")
    add("")

    report = "\n".join(L)
    out_path = out_path or (results_dir / f"{stem}.report.md")
    out_path.write_text(report)

    lattice_json = results_dir / f"{stem}.gate_failure_probe.json"
    lattice_json.write_text(json.dumps({
        "source_artifact": str(artifact_path),
        "total_three_point_gate_failed": total_gate,
        "answer_ambiguity_share": gate_cause["answer_ambiguity"],
        "orig_ne_rep_cases": lattice["n_cases"],
        "note": "three_point_gate_failed diagnostics. `answer_ambiguity_share` = failures where the "
                "subject has >=2 competing same-type answers (dominant cause). "
                "`lattice_probe_hits` = the build's locator probe; `level_kept_in_doc=true` means the "
                "flagged level is used by a KEPT relation -> FALSE positive, NOT a lattice-data defect. "
                "`suspects` = every generalised argument in an orig!=rep case (unconfirmed).",
        "lattice_probe_hits": lattice["tier1"],
        "suspects": lattice["tier2"],
    }, indent=1))

    quality_json = results_dir / f"{stem}.lattice_quality.json"
    quality_json.write_text(json.dumps({
        "source_artifact": str(artifact_path),
        "note": "Malformed generalization level strings in lattice_profiles.json, audited directly "
                "(independent of the reader). Fix these entries: grouping-node-artifact ('... family'), "
                "snomed-axis-label (hierarchy axis descriptors, not concepts), admin-boilerplate, "
                "fsn-verbose, garbled-surface-key. `max_used`/`max_kept` = selection frequency.",
        "flagged_entities": lattice_quality,
    }, indent=1))
    print(f"wrote {quality_json}")

    print(f"wrote {out_path}")
    print(f"wrote {lattice_json}")
    print(f"consistency self-check: {consistency}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="build directory containing <stem>.utility.json")
    ap.add_argument("--out", type=Path, default=None, help="report path (default <stem>.report.md in the dir)")
    args = ap.parse_args()
    if not args.results_dir.is_dir():
        sys.exit(f"not a directory: {args.results_dir}")
    build_report(args.results_dir, args.out)


if __name__ == "__main__":
    main()
