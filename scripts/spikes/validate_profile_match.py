"""Design-validation spike for the retrieve-then-verify profile matcher.

Validates docs/specs/substitutor-profile-match-retrieve-verify.md end-to-end with the REAL
bge-small embedder and the REAL DeBERTa NLI gate against the real medical profile artifact.
Measures whether the design recovers the documented exact-match failure modes (plural, article,
punctuation, typo, morphology, modifier, synonym) and abstains correctly on negatives.

One-off spike (scripts/spikes/). CPU only. Thresholds stay at spec values (no tuning).

Instrumentation note: proposer recall needs the FULL top-k candidate list, but
match_profile_entry stops NLI at the first approval and never exposes the list. So this spike
reproduces the retrieval step (lines mirror profile_match.match_profile_entry steps 3) using the
same index + embed_fn to record candidates, then calls the real match_profile_entry unchanged for
the certified verdict. Retrieval-miss vs NLI-refusal vs wrong-entry-won is separable from the two.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import time
from pathlib import Path

import numpy as np

from cloak import profile_match as pm
from cloak.lattice import nli_gate

PROFILES = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.json")
INDEX = Path(os.environ.get("SPIKE_INDEX", "/tmp/claude-1000/spike_profile_match.embindex.npz"))

# (query_surface, runtime_type, context_sentence [contains query verbatim], gold | "ABSTAIN", failure_mode)
POSITIVES = [
    ("heart murmurs", "health-condition", "The exam detected heart murmurs on auscultation.", "heart murmur", "plural"),
    ("heart problem", "health-condition", "He has a long history of heart problem.", "heart problems", "plural"),
    ("aspirins", "drug", "The bottle contained several aspirins.", "aspirin", "plural"),
    ("a cataract", "health-condition", "The surgeon removed a cataract from the left eye.", "cataract", "article"),
    ("an aspirin", "drug", "The nurse gave the patient an aspirin.", "aspirin", "article"),
    ("diabetes.", "health-condition", "The patient was diagnosed with diabetes.", "diabetes mellitus", "punct"),
    ("metformin.", "drug", "Each morning the patient takes metformin.", "metformin", "punct"),
    ("aagenes syndrome", "health-condition", "Genetic testing confirmed aagenes syndrome in the infant.", "aagenaes syndrome", "typo"),
    ("ibuprofin", "drug", "She took ibuprofin for the headache.", "ibuprofen", "typo"),
    ("diabetic", "health-condition", "The patient is diabetic.", "diabetes mellitus", "morphology"),
    ("hypertensive", "health-condition", "The patient is hypertensive.", "hypertension", "morphology"),
    ("severe asthma", "health-condition", "The child has severe asthma.", "asthma", "modifier"),
    ("intravenous morphine", "drug", "The patient received intravenous morphine post-operatively.", "morphine", "modifier"),
    ("belly scan", "health-condition", "The radiologist ordered a belly scan.", "abdominal ct", "synonym"),
    ("blood thinner", "drug", "The cardiologist started him on a blood thinner.", "warfarin", "synonym"),
]

NEGATIVES = [
    ("zxqwvbn plooghor", "health-condition", "The note listed zxqwvbn plooghor as a finding.", "ABSTAIN", "gibberish"),
    ("ibuprofen", "health-condition", "The chart recorded ibuprofen under conditions.", "ABSTAIN", "wrong-type"),
    ("suspected cancer", "health-condition", "The referral mentions suspected cancer.", "ABSTAIN", "non-entailed-modifier"),
    ("possible fracture", "health-condition", "X-ray was ordered for a possible fracture.", "ABSTAIN", "non-entailed-modifier"),
    ("malaria", "health-condition", "The traveler returned with malaria.", "ABSTAIN", "novel-absent"),
]


def main():
    # shared embedder with a per-text cache so a query is embedded once (accurate timing).
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(pm.DEFAULT_MODEL_ID)
    cache: dict[str, np.ndarray] = {}
    stats = {"embed_time": 0.0, "embed_calls": 0, "nli_time": 0.0, "nli_pairs": 0}

    def embed_fn(texts):
        miss = [t for t in texts if t not in cache]
        if miss:
            t0 = time.perf_counter()
            vecs = np.asarray(model.encode(miss, normalize_embeddings=True), dtype=np.float32)
            stats["embed_time"] += time.perf_counter() - t0
            stats["embed_calls"] += len(miss)
            for t, v in zip(miss, vecs):
                cache[t] = v
        return np.stack([cache[t] for t in texts])

    def nli_fn(entity, ctx, levels):
        stats["nli_pairs"] += len(levels)  # upper bound: gate drops self-referential/degenerate
        t0 = time.perf_counter()
        out = nli_gate(entity, ctx, levels, thresh=pm.NLI_THRESH)
        stats["nli_time"] += time.perf_counter() - t0
        return out

    # --- build index (dominant cost = embedding the whole corpus) ---
    t0 = time.perf_counter()
    pm.build_embindex(PROFILES, out_path=INDEX, embed_fn=embed_fn)
    build_time = time.perf_counter() - t0
    corpus_embed_time = stats["embed_time"]
    corpus_embed_calls = stats["embed_calls"]

    index = pm.load_embindex(str(INDEX), str(PROFILES))
    assert index is not None, "index failed to load (hash/schema mismatch?)"

    def retrieve(span_text, runtime_type):
        """Mirror profile_match retrieval (step 3) to expose the candidate list for recall."""
        idxs = index.type_rows(runtime_type)
        if not idxs:
            return []
        q = pm._l2norm(embed_fn([span_text]))[0]
        sims = index.vectors[idxs] @ q
        kept = [(idxs[p], float(sims[p])) for p in np.argsort(-sims) if sims[p] >= pm.SIM_FLOOR][:pm.TOP_K]
        best: dict[str, float] = {}
        for row_i, sim in kept:
            c = index.rows[row_i]["canonical"]
            best[c] = max(best.get(c, -1.0), sim)
        return sorted(best.items(), key=lambda kv: -kv[1])

    rows = []
    for query, rtype, ctx, gold, mode in POSITIVES + NEGATIVES:
        cand = retrieve(query, rtype)
        m = pm.match_profile_entry(query, rtype, ctx, profiles_path=PROFILES,
                                   index_path=INDEX, embed_fn=embed_fn, nli_fn=nli_fn)
        is_pos = gold != "ABSTAIN"
        matched = m.entry if m else None
        if is_pos:
            passed = m is not None and matched == gold
        else:
            passed = m is None
        rows.append({
            "query": query, "type": rtype, "gold": gold, "mode": mode, "is_pos": is_pos,
            "matched": matched, "sim": m.similarity if m else (cand[0][1] if cand else 0.0),
            "kind": m.kind if m else "-", "n_levels": len(m.levels) if m else 0,
            "pass": passed, "cand": cand,
        })

    # --- table ---
    hdr = f"{'query':22} {'type':16} {'gold':22} {'matched':22} {'sim':>5} {'kind':8} {'nlv':>3} {'result':6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['query'][:22]:22} {r['type']:16} {str(r['gold'])[:22]:22} "
              f"{str(r['matched'])[:22]:22} {r['sim']:5.3f} {r['kind']:8} {r['n_levels']:>3} "
              f"{'PASS' if r['pass'] else 'FAIL':6}")

    # --- per-case retrieval diagnostic (candidate lists) ---
    print("\nRETRIEVAL CANDIDATES (canonical@sim, top-k>=SIM_FLOOR; empty = retrieval miss):")
    for r in rows:
        cl = ", ".join(f"{c}@{s:.3f}" for c, s in r["cand"]) or "(none above floor)"
        gold_in = "" if not r["is_pos"] else ("  [gold retrieved]" if any(c == r["gold"] for c, _ in r["cand"]) else "  [GOLD MISSED]")
        print(f"  {r['query'][:22]:22} {r['mode']:22} -> {cl}{gold_in}")

    # --- summary metrics ---
    pos = [r for r in rows if r["is_pos"]]
    neg = [r for r in rows if not r["is_pos"]]
    recall_hits = [r for r in pos if any(c == r["gold"] for c, _ in r["cand"])]
    returned = [r for r in rows if r["matched"] is not None]
    correct_returned = [r for r in returned if r["is_pos"] and r["matched"] == r["gold"]]
    pos_pass = [r for r in pos if r["pass"]]
    neg_pass = [r for r in neg if r["pass"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"positives: {len(pos)}   negatives: {len(neg)}")
    print(f"proposer recall (gold in top-k >= {pm.SIM_FLOOR}): "
          f"{len(recall_hits)}/{len(pos)} = {len(recall_hits) / len(pos):.2f}")
    print(f"certified precision (of returned matches, frac == gold): "
          f"{len(correct_returned)}/{len(returned)} = {len(correct_returned) / len(returned):.2f}"
          if returned else "certified precision: n/a (nothing returned)")
    print(f"positive end-to-end pass (returned == gold): "
          f"{len(pos_pass)}/{len(pos)} = {len(pos_pass) / len(pos):.2f}")
    print(f"abstain correctness on negatives: "
          f"{len(neg_pass)}/{len(neg)} = {len(neg_pass) / len(neg):.2f}")

    # per-mode breakdown
    print("\nper-failure-mode (positives): pass / total")
    modes: dict[str, list] = {}
    for r in pos:
        modes.setdefault(r["mode"], []).append(r)
    for mode, rs in modes.items():
        types = sorted({r["type"] for r in rs})
        print(f"  {mode:16} {sum(r['pass'] for r in rs)}/{len(rs)}  types={types}")

    print("\nTIMINGS")
    print(f"  index build (incl. corpus embed): {build_time:.2f}s  "
          f"(corpus embed {corpus_embed_time:.2f}s for {corpus_embed_calls} surfaces)")
    query_embed = stats["embed_time"] - corpus_embed_time
    query_calls = stats["embed_calls"] - corpus_embed_calls
    print(f"  query embedding: {query_embed:.2f}s for {query_calls} unique queries")
    print(f"  NLI: {stats['nli_time']:.2f}s over ~{stats['nli_pairs']} hypotheses "
          f"(pairs counted before gate's self-ref/degenerate drop)")


if __name__ == "__main__":
    main()
