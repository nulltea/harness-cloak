"""Reprocess an existing lattice_profiles.json: merge synonym rows (entity merge), validate,
rewrite atomically, rebuild the embedding index.

Spec: docs/specs/lattice-entry-dedup-and-span-resolution.md (Part 3, reprocess CLI). The gate
is enabled only when --gate-eval points to a calibration artifact whose chosen_threshold is
non-null (scripts/calibrate_entity_merge_gate.py). Merged-pair listings go to the JSON report,
not stdout.

  .venv/bin/python scripts/dedupe_lattice_profile_entries.py \
      --gate-eval results/entity_merge_gate_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloak.lattice_producer.entity_merge import DEFAULT_OBO_PATHS, apply_entity_merge
from cloak.lattice_profiles import DEFAULT_PROFILE_PATH, validate_profile_artifact


def _atomic_write(path: Path, artifact: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default=str(DEFAULT_PROFILE_PATH))
    ap.add_argument("--gate-eval", default=None,
                    help="calibration JSON; gate enabled iff chosen_threshold is non-null")
    ap.add_argument("--obo", action="append", default=[],
                    help="runtime_type=obo_path override (default: health-condition=DOID)")
    ap.add_argument("--report-out", default="results/entity_merge_report.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-embindex", action="store_true")
    ap.add_argument("--no-embed-blocking", action="store_true",
                    help="block on identical levels only (no embedding model)")
    args = ap.parse_args(argv)

    profiles_path = Path(args.profiles)
    artifact = json.loads(profiles_path.read_text())

    obo_paths = dict(DEFAULT_OBO_PATHS)
    for spec in args.obo:
        runtime_type, _, obo = spec.partition("=")
        obo_paths[runtime_type] = obo

    gate_fn = gate_threshold = None
    if args.gate_eval:
        eval_art = json.loads(Path(args.gate_eval).read_text())
        gate_threshold = eval_art.get("chosen_threshold")
        if gate_threshold is not None:
            from calibrate_entity_merge_gate import nli_gate_scorer

            score_pairs = nli_gate_scorer(eval_art["model_id"])
            gate_fn = lambda sa, sb: max(score_pairs([(a, b) for a in sa for b in sb]))
            print(f"gate enabled: {eval_art['model_id']} @ {gate_threshold}", flush=True)
        else:
            print("gate eval has chosen_threshold=null -> gate disabled (review-only)",
                  flush=True)

    embed_fn = None
    if not args.no_embed_blocking:
        from cloak.profile_match import DEFAULT_MODEL_ID, _st_model

        model = _st_model(DEFAULT_MODEL_ID)
        embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True)

    before = {rt: len(entries) for rt, entries in artifact.get("profiles", {}).items()}
    report = apply_entity_merge(artifact, obo_paths=obo_paths, gate_fn=gate_fn,
                                gate_threshold=gate_threshold, embed_fn=embed_fn)
    after = {rt: len(entries) for rt, entries in artifact.get("profiles", {}).items()}
    report["entry_counts"] = {rt: {"before": before[rt], "after": after[rt]}
                              for rt in before}

    errors = validate_profile_artifact(artifact)
    if errors:
        sys.exit("post-merge validation failed:\n" + "\n".join(errors[:20]))

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2, sort_keys=True))
    merged_n = sum(len(t["merged"]) for t in report.get("types", {}).values())
    review_n = sum(len(t["review"]) for t in report.get("types", {}).values())
    print(f"merged pairs: {merged_n}; review pairs: {review_n}; "
          f"entry counts: {json.dumps(report['entry_counts'])}", flush=True)
    if args.dry_run:
        print("dry-run: artifact NOT written", flush=True)
        return
    _atomic_write(profiles_path, artifact)
    if not args.skip_embindex:
        from cloak.profile_match import build_embindex

        out = build_embindex(profiles_path)
        print(f"embindex rebuilt: {out}", flush=True)


if __name__ == "__main__":
    main()
