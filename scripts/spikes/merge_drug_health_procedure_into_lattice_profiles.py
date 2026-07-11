"""One-off: merge the nemotron-3-super drug-health-procedure proposal into
data/lattice_profiles/lattice_profiles.json.

lattice_profiles.json is itself a build output of scripts/merge_lattice_profiles.py (from
comm_lattice_profiles.json + mined_lattice_profiles.json) -- this reuses the same
merge_profile_artifacts() logic (now level_counts/level_grounding aware) rather than hand-editing
the JSON, so the result stays structurally consistent with what that script produces.

It also reschematizes the pre-existing (non-drug/health/procedure) profiles to the proposal's
entry schema by stamping entry_origin="observed-surface" on every base row -- these rows are real
dataset surfaces (GeoNames/CLDR/ESCO/...), so "observed-surface" is factual. level_counts and
level_grounding are deliberately NOT synthesized for them: those are per-level anonymity-set sizes
that only a producer run grounds, and fabricating them from the single top-level `count` would
invent privacy numbers (empirical-honesty rule). merge_profile_artifacts() preserves untouched
profiles verbatim, so the stamped entry_origin flows through; the 3 incoming profiles are deep-copied
with their own entry_origin/level_counts/level_grounding intact.

The incoming runtime types (drug/health-condition/medical-procedure) REPLACE, not union with, any
same-typed profiles already in the base: a regenerated proposal supersedes the prior run's rows, so
we drop those types from the base before merging. This makes the merge idempotent whether the base
is a fresh 9-profile prep or already carries a previous drug/health/procedure merge.

With --ontology-dedup, the incoming types are unioned (no alias fold) and then run through
ontology-gated entity dedup (cloak.lattice_producer.entity_merge.apply_entity_merge): a DOID
oracle for health-condition plus a precision-calibrated NLI gate (results/entity_merge_gate_eval.json).
Human-reviewed merges from a --curated JSON ([runtime_type, keep, fold] rows) are applied afterward.

Usage:
  python <this> [incoming.json]                       # default alias-fold merge (unchanged)
  python <this> --ontology-dedup [--curated merges.json] [--dry-run] [incoming.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from merge_lattice_profiles import apply_curated_merges, merge_profile_artifacts  # noqa: E402

BASE = Path("data/lattice_profiles/lattice_profiles.json")
INCOMING = Path("data/lattice_profiles/proposed/drug-health-procedure-nemotron-3-super.proposed.json")
GATE_EVAL = Path("results/entity_merge_gate_eval.json")
REVIEW_OUT = Path("results/lattice_merge_entity_review.json")


def _ontology_gated_dedup(merged: dict) -> None:
    """Ontology-gated entity dedup over the un-deduped union, in place. Gate scoped to the DOID
    oracle types (DEFAULT_OBO_PATHS); disabled if the calibration chosen_threshold is null. Writes
    the review report and prints per-type merged/review counts only (no term listings)."""
    from cloak.lattice_producer.entity_merge import DEFAULT_OBO_PATHS, apply_entity_merge
    from cloak.profile_match import DEFAULT_MODEL_ID, _st_model

    gate_fn = gate_threshold = None
    if GATE_EVAL.exists():
        eval_art = json.loads(GATE_EVAL.read_text())
        gate_threshold = eval_art.get("chosen_threshold")
        if gate_threshold is not None:
            from calibrate_entity_merge_gate import nli_gate_scorer

            score_pairs = nli_gate_scorer(eval_art["model_id"])
            gate_fn = lambda sa, sb: max(score_pairs([(a, b) for a in sa for b in sb]))
            print(f"gate enabled: {eval_art['model_id']} @ {gate_threshold}", flush=True)
        else:
            print("gate eval chosen_threshold=null -> gate disabled (review-only)", flush=True)
    else:
        print(f"no gate eval at {GATE_EVAL} -> gate disabled (review-only)", flush=True)

    model = _st_model(DEFAULT_MODEL_ID)
    embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True)

    report = apply_entity_merge(merged, obo_paths=DEFAULT_OBO_PATHS, gate_fn=gate_fn,
                                gate_threshold=gate_threshold, embed_fn=embed_fn)
    REVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote entity-merge review report: {REVIEW_OUT}", flush=True)
    for rt, t in sorted(report.get("types", {}).items()):
        print(f"  {rt}: merged={len(t['merged'])} review={len(t['review'])} "
              f"gate_scored={t['gate_scored']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("incoming", nargs="?", default=str(INCOMING))
    ap.add_argument("--ontology-dedup", action="store_true",
                    help="union incoming (no alias fold) then run ontology-gated entity dedup")
    ap.add_argument("--curated", default=None,
                    help="JSON list of [runtime_type, keep, fold] merges applied post-dedup")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except write lattice_profiles.json")
    args = ap.parse_args()

    base = json.loads(BASE.read_text())
    incoming = json.loads(Path(args.incoming).read_text())

    dropped = {rt: len(base["profiles"].get(rt, {})) for rt in incoming.get("profiles", {}) if rt in base.get("profiles", {})}
    for rt in incoming.get("profiles", {}):
        base.get("profiles", {}).pop(rt, None)
    print(f"dropped superseded base profiles: {dropped}")

    stamped = 0
    for entries in base.get("profiles", {}).values():
        for row in entries.values():
            if "entry_origin" not in row:
                row["entry_origin"] = "observed-surface"
                stamped += 1
    print(f"stamped entry_origin on {stamped} existing rows")

    if args.ontology_dedup:
        merged = merge_profile_artifacts(base, incoming, entity_dedup=True)
        _ontology_gated_dedup(merged)
        if args.curated:
            pairs = [tuple(p) for p in json.loads(Path(args.curated).read_text())]
            apply_curated_merges(merged, pairs)
            print(f"applied {len(pairs)} curated merge pair(s) from {args.curated}", flush=True)
        from cloak.lattice_profiles import validate_profile_artifact

        errors = validate_profile_artifact(merged)
        if errors:
            sys.exit("post-dedup validation failed:\n" + "\n".join(errors[:20]))
    else:
        merged = merge_profile_artifacts(base, incoming)

    if args.dry_run:
        print("dry-run: lattice_profiles.json NOT written", flush=True)
    else:
        BASE.write_text(json.dumps(merged, indent=2, sort_keys=True))
        print(f"wrote {BASE}")

    with_counts = {
        rt: sum(1 for row in entries.values() if "level_counts" in row)
        for rt, entries in merged["profiles"].items()
        if rt in ("drug", "health-condition")
    }
    print(f"profiles: {{{', '.join(f'{k}: {len(v)}' for k, v in merged['profiles'].items())}}}")
    print(f"entries with level_counts: {with_counts}")


if __name__ == "__main__":
    main()
