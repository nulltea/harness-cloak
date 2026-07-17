"""Re-score an existing QA-v2 build with the current pinned reader — no teacher call.

Replays the teacher proposals already recorded in a prior artifact's
`relation_generation`/`relation_candidate_accounting` (so OpenRouter is never
hit) and re-runs the whole compile + three-point gate against whatever
`BatchedContextReader` is pinned right now. Used to test a reader change in
isolation. Patches the threshold manifest's reader_pin to the current default
so the build's pin-consistency check passes.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import build_qa_utility_artifact as qa_cli
from cloak.train.qa_builder import (
    DEFAULT_CONTEXT_READER_PIN,
    RelationTeacherProposals,
    read_context_batch,
)

_PROPOSAL_FIELDS = ("relation", "arguments", "question", "accepted_answers", "scoring_contract")


class OfflineTeacher:
    """Replays cached proposals for exactly the recorded documents."""

    def __init__(self, prior_artifact: dict):
        self.pin = dict(prior_artifact["teacher_pin"])
        self._relations = prior_artifact["relation_generation"]
        self._accounting = prior_artifact["relation_candidate_accounting"]
        self._doc_order = list(self._relations)
        self._call = 0

    def propose(self, prompt, *, response_format=None):
        # The build iterates documents in a stable order and calls once per
        # under-supported doc; map calls to that order.
        doc_id = self._doc_order[self._call]
        self._call += 1
        relations = [
            {k: row.get(k) for k in _PROPOSAL_FIELDS}
            for row in self._relations[doc_id]
        ]
        return RelationTeacherProposals(relations, self._accounting[doc_id])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="/tmp/qa-v2-d2n002-lattice.json")
    ap.add_argument("--env", default="/tmp/ranker_env_qa_v2_d2n002_gated.json")
    ap.add_argument("--arms", default="/tmp/task_arms_qa_v2_d2n002_gated.json")
    ap.add_argument("--manifest", default="/tmp/qa-v2-d2n002-manifest.json")
    ap.add_argument("--corpus", default="clinical")
    ap.add_argument("--doc-id", action="append", default=None)
    ap.add_argument("--out", default="/tmp/qa-v2-d2n002-reader-v2.json")
    args = ap.parse_args()

    if not os.getenv("CLOAK_LLM_CACHE"):
        os.environ["CLOAK_LLM_CACHE"] = "/tmp/"

    prior = json.load(open(args.prior))
    doc_ids = args.doc_id or list(prior["relation_generation"])

    # patch manifest reader_pin -> current default so the pin check passes
    manifest = json.load(open(args.manifest))
    manifest["reader_pin"] = dict(DEFAULT_CONTEXT_READER_PIN)
    patched_manifest = args.manifest + ".readerv2.json"
    json.dump(manifest, open(patched_manifest, "w"), indent=2)

    build_args = SimpleNamespace(
        env=args.env,
        arms=args.arms,
        corpus=args.corpus,
        doc_id=doc_ids,
        threshold_manifest=patched_manifest,
        out=args.out,
        relation_teacher=False,
    )
    artifact = qa_cli.build_from_files(
        build_args, relation_teacher=OfflineTeacher(prior), reader=read_context_batch
    )
    json.dump(artifact, open(args.out, "w"), indent=2)

    # summary
    assertions = artifact["assertions"]
    rejections = artifact["rejections"]
    n_assert = sum(len(v) for v in assertions.values())
    print(f"wrote {args.out}: assertions={n_assert}")
    for doc_id in doc_ids:
        gen = artifact["relation_generation"].get(doc_id, [])
        kept = [g for g in gen if g.get("status") == "kept"]
        print(f"\n== {doc_id}: {len(kept)}/{len(gen)} relations kept ==")
        for g in gen:
            print(f"  [{g.get('status'):8}] {g.get('relation'):24} "
                  f"{g.get('reason','')[:60]}  Q={g.get('question','')[:70]!r}")


if __name__ == "__main__":
    main()
