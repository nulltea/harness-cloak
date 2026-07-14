"""Ablation: run the production build with the HOW TO INSPECT THE SOURCE section
stripped from the teacher prompt, on the old-detector env, to see which
deterministic gates catch the relations that section's guidance pre-empts
(multi-turn scope + the conditional/hedge caution).

Production code is untouched: the prompt builder is monkeypatched in-process.
One live teacher call + reader calls to the pinned local reader.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import cloak.train.qa_builder as qb

_original_prompt = qb.relation_teacher_prompt


def _ablated_prompt(doc_id, document, environment_document):
    full = _original_prompt(doc_id, document, environment_document)
    start = full.index("HOW TO INSPECT THE SOURCE")
    end = full.index("PRIVACY-SAFE QA")
    return full[:start] + full[end:]


qb.relation_teacher_prompt = _ablated_prompt

import build_qa_utility_artifact as builder


def main():
    args = builder.parse_args([
        "--env", "/tmp/ranker_env_qa_v2_d2n002.json",
        "--arms", "/tmp/task_arms_qa_v2_d2n002.json",
        "--corpus", "clinical", "--doc-id", "aci/D2N002",
        "--threshold-manifest", "/tmp/qa-v2-d2n002-manifest.json",
        "--out", "/tmp/qa-v2-d2n002-ablated-nohowto.json",
        "--relation-teacher",
    ])
    artifact = builder.build_from_files(args, relation_teacher=qb.OpenRouterRelationTeacher())
    out = args.out
    json.dump(artifact, open(out, "w"), indent=1)
    print(f"wrote {out}", flush=True)

    attempts = artifact.get("relation_generation", {}).get("aci/D2N002", [])
    print(f"\nteacher attempts: {len(attempts)}", flush=True)
    for a in attempts:
        kinds = "+".join(x.get("kind", "?") for x in a.get("arguments") or [])
        print(f"  [{a['proposal_index']}] {a.get('relation'):26} {kinds:16} "
              f"{a.get('status'):8} {a.get('reason','')}", flush=True)
    from collections import Counter
    recs = [r["detail_reason"] for r in artifact["rejections"]["records"]
            if r.get("evidence", {}).get("source") in {"relation_teacher", "context_validation"}]
    print("\nrelation rejection reasons:", dict(Counter(recs)), flush=True)
    kept = [r for r in artifact["assertions"].values() if r.get("subtype") == "contextual_relation"]
    print("kept contextual relations:", len(kept), flush=True)


if __name__ == "__main__":
    main()
