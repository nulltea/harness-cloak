"""Free check that deterministic-stage keeps survive the post-gleaning kept-set rebuild.

Runs the production builder with the REAL cached primary teacher and an ABSTAINING stub as the
gleaning secondary (0 paid calls): the escalation/gleaning code path executes end-to-end (targets,
batches, merge, `accepted` rebuild), so the rebuild regression -- stage keeps dropped by
`accepted = pre_teacher_accepted + merged_relations` -- reproduces or is proven fixed.

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
        scripts/spikes/stage_gleaning_rebuild_check.py
"""
from __future__ import annotations

import collections

import build_qa_utility_artifact as bqa

DOC_IDS = ["aci/D2N002", "aci/D2N008", "aci/D2N009", "aci/D2N011", "aci/D2N025"]


class AbstainingTeacher:
    pin = {"model": "abstaining-stub", "provider": None}

    def propose(self, prompt, response_format=None):
        return []


def main():
    args = bqa.parse_args([
        "--env", "results/qa_v2_aci_full/ranker-env.json",
        "--arms", "results/qa_v2_aci_full/arms.json",
        "--threshold-manifest", "data/qa_v2/relation_gate_manifest.json",
        *[x for doc in DOC_IDS for x in ("--doc-id", doc)],
        "--relation-teacher", "--relation-support-prefilter", "--relation-deterministic-stage",
        "--relation-teacher-gleaning",
        "--out", "/tmp/unused",
    ])
    artifact = bqa.build_from_files(args, secondary_relation_teacher=AbstainingTeacher())
    runs = collections.Counter()
    for row in artifact["assertions"].values():
        if row.get("subtype") == "contextual_relation":
            runs[(row.get("evidence") or {}).get("run_id")] += 1
    print("kept relations by run_id (gleaning path, abstaining secondary):", dict(runs))
    stage_expected = sum(
        (e.get("deterministic_stage") or {}).get("kept_count") or 0
        for e in (artifact.get("relation_escalation") or {}).values())
    print(f"stage kept_count accounting: {stage_expected} | "
          f"stage rows in accepted: {runs.get('deterministic_stage', 0)}")
    assert runs.get("deterministic_stage", 0) == stage_expected, "stage keeps dropped by rebuild!"
    targets = sum((e.get("gleaning") or {}).get("target_count") or 0
                  for e in (artifact.get("relation_escalation") or {}).values())
    print(f"gleaning targets sent (post proposed-keys fix): {targets}")
    print("OK: stage keeps survive the post-gleaning rebuild")


if __name__ == "__main__":
    main()
