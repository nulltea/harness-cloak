"""End-to-end smoke for the profile-match substitutor pre-pass, on REAL models.

Drives cloak.substitute.substitute() over hand-built Spans (no Detector) against the proposed
drug + health-condition artifact, proving the whole path lives:
retrieve (bge-small) -> certify (DeBERTa NLI) -> lattice_for(proposal=...) -> R match provenance.

CPU only (models are in the local HF cache). Not a privacy calibration: walk_risk pools are absent
for these fine types (-> risk 1.0), so tau=2.0 lets every certified generalization ship; the smoke
validates plumbing, not tau. One-off spike (scripts/spikes/).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU BEFORE any torch/transformers/cloak import
os.environ["HIP_VISIBLE_DEVICES"] = ""

import functools
from pathlib import Path

from cloak import substitute as sub_mod
from cloak.detect import Span
from cloak.profile_match import build_embindex, match_spans_batch
from cloak.substitute import substitute

PROFILES = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.json")
INDEX = Path("/tmp/claude-1000/smoke_substitute_integration.embindex.npz")

# document + spans: one exact hit (canonical "aspirin"), two semantic variants (morphology,
# plural), one nonsense surface that must abstain to a typed placeholder.
DOC = ("The patient is diabetic and takes aspirin daily. "
       "Auscultation revealed heart murmurs. "
       "The note also listed zxqwvbn plooghor as a finding.")
SPAN_SPECS = [
    ("diabetic", "health-condition"),          # semantic (morphology -> diabetes mellitus)
    ("aspirin", "drug"),                        # exact (canonical surface)
    ("heart murmurs", "health-condition"),      # semantic (plural -> heart murmur)
    ("zxqwvbn plooghor", "health-condition"),   # abstain -> placeholder
]


def make_spans():
    spans = []
    for surface, rtype in SPAN_SPECS:
        start = DOC.index(surface)
        spans.append(Span(start=start, end=start + len(surface), text=surface,
                          type=rtype, score=1.0, source="smoke"))
    return spans


def main():
    print(f"building throwaway index for {PROFILES} -> {INDEX}")
    build_embindex(PROFILES, out_path=INDEX)   # real bge-small, CPU

    # smallest override: point substitute()'s pre-pass at the proposed artifact + tmp index.
    sub_mod.match_spans_batch = functools.partial(
        match_spans_batch, profiles_path=PROFILES, index_path=INDEX)

    doc_p, R = substitute(DOC, make_spans(), tau=2.0)

    print("\ndoc_orig:", DOC)
    print("doc_p   :", doc_p)
    print("\nR (substitution record):")
    for r in R:
        print(f"  {r['action']:11s} {r['surface']!r:20s} -> {r['replacement']!r:26s} "
              f"match={r.get('match')}")

    by_surface = {r["surface"]: r for r in R}

    # 1. at least one semantic match with entry + nli populated flowed into a replacement
    semantic = [r for r in R if r.get("match", {}).get("kind") == "semantic"]
    assert semantic, "no semantic match reached R"
    for r in semantic:
        m = r["match"]
        assert m["entry"], f"semantic match missing entry: {r}"
        assert m["nli"] is not None, f"semantic match missing nli score: {r}"
        assert r["action"] == "generalize", f"semantic hit did not generalize: {r}"
    assert any(r["surface"] in ("diabetic", "heart murmurs") for r in semantic), \
        f"expected a variant surface to match semantically, got {[r['surface'] for r in semantic]}"

    # 2. exact hit carries {"kind": "exact"}
    aspirin = by_surface["aspirin"]
    assert aspirin.get("match") == {"kind": "exact"}, f"aspirin not an exact hit: {aspirin}"

    # 3. nonsense surface abstains to a typed placeholder
    nonsense = by_surface["zxqwvbn plooghor"]
    assert nonsense["action"] == "placeholder", f"nonsense did not abstain: {nonsense}"
    assert "match" not in nonsense, f"abstained span should carry no match block: {nonsense}"
    assert "zxqwvbn" not in doc_p, "nonsense surface leaked into doc_p"

    print("\nsmoke_substitute_integration OK: semantic + exact + abstain all verified")


if __name__ == "__main__":
    main()
