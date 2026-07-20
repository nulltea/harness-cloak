"""Deterministic-only coverage probe for the relation stage (throwaway spike).

Question: how much of what the TEACHER keeps (primary + gleaning) could the deterministic stage
have produced alone? Runs the full production builder with a NULL teacher (abstains on every doc)
and --relation-deterministic-stage semantics, so the stage generates with an EMPTY kept-set through
the real compile + reader gate. Then compares kept relation facts against a teacher-driven
reference artifact (the stage-off A/B arm) at the decision-level fact-key granularity
(compound literal rows decomposed into per-literal pairs).

Zero teacher calls; reader/judge = local MedGemma. One GPU process.

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
        scripts/spikes/deterministic_only_stage_coverage.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from cloak.train.qa_builder import _relation_fact_key

import build_qa_utility_artifact as bqa

DOC_IDS = ["aci/D2N002", "aci/D2N008", "aci/D2N009", "aci/D2N011", "aci/D2N025"]
REFERENCE = "results/qa_v2_stage_ab/off"   # stage-off A/B arm's utility artifact (teacher-driven)
TEACHER_REFERENCE_RUNS = {"primary", "gleaning"}
OUT = Path("results/qa_v2_stage_ab/deterministic_only")


class NullRelationTeacher:
    """Abstains on every prompt: primary keeps = empty, so the stage sees an empty kept-set."""
    pin = {"model": "null-teacher", "provider": None}

    def propose(self, prompt, response_format=None):
        return []


def kept_pair_keys(artifact: dict, run_ids: set[str] | None) -> set[tuple]:
    """Decision-level (relation, subject-identity, object-identity) pair keys of kept relation
    rows; a compound literal row contributes one key per context literal."""
    keys: set[tuple] = set()
    for row in artifact["assertions"].values():
        if row.get("subtype") != "contextual_relation":
            continue
        evidence = row.get("evidence") or {}
        if run_ids is not None and evidence.get("run_id") not in run_ids:
            continue
        occurrences = {
            str(o["occurrence_id"]): o
            for o in artifact["documents"][row["doc_id"]]["occurrences"]
        }
        arguments = list(evidence.get("arguments") or [])
        linked = [a for a in arguments if a.get("kind") == "linked"]
        contexts = [a for a in arguments if a.get("kind") == "context"]
        try:
            if len(arguments) == 2:
                keys.add(_relation_fact_key(str(row.get("relation")), arguments, occurrences))
            elif len(linked) == 1 and contexts:  # compound literal row -> per-literal pairs
                for context in contexts:
                    keys.add(_relation_fact_key(
                        str(row.get("relation")), [linked[0], context], occurrences))
        except ValueError:
            continue
    return keys


def main():
    args = bqa.parse_args([
        "--env", "results/qa_v2_aci_full/ranker-env.json",
        "--arms", "results/qa_v2_aci_full/arms.json",
        "--threshold-manifest", "data/qa_v2/relation_gate_manifest.json",
        *[x for doc in DOC_IDS for x in ("--doc-id", doc)],
        "--relation-teacher", "--relation-support-prefilter", "--relation-deterministic-stage",
        "--out", str(OUT / "det_only"),
    ])
    artifact = bqa.build_from_files(args, relation_teacher=NullRelationTeacher())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "det_only.utility.json").write_text(json.dumps(artifact, sort_keys=True))

    runs = collections.Counter()
    for row in artifact["assertions"].values():
        if row.get("subtype") == "contextual_relation":
            runs[(row.get("evidence") or {}).get("run_id")] += 1
    print("deterministic-only kept relations by run_id:", dict(runs))

    reference = json.loads(Path(REFERENCE).read_text())
    teacher_keys = kept_pair_keys(reference, TEACHER_REFERENCE_RUNS)
    reference_all = kept_pair_keys(reference, None)
    det_keys = kept_pair_keys(artifact, None)
    print(f"\nteacher-kept facts (primary+gleaning in reference): {len(teacher_keys)}")
    print(f"deterministic-only kept facts:                       {len(det_keys)}")
    print(f"teacher facts COVERED by deterministic-only:         "
          f"{len(teacher_keys & det_keys)}/{len(teacher_keys)}")
    print(f"deterministic-only EXTRA facts (not teacher-kept):   {len(det_keys - teacher_keys)}")
    print(f"reference-arm ALL kept facts (any run):              {len(reference_all)} "
          f"(covered: {len(reference_all & det_keys)})")
    missed = collections.Counter(k[0] for k in (teacher_keys - det_keys))
    print("teacher facts MISSED by deterministic-only, by relation:", dict(missed))


if __name__ == "__main__":
    main()
