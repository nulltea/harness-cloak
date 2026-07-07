"""Environment diagnostics before surrogate-RL training (docs/plans/2026-07-03-surrogate-rl-gaps-fixes.md).

Three measurements on the cached constructed-arms round trips (no new remote calls):

  absent      - decompose gen_absent: exact echo / fuzzy/semantic-invertible today /
                loosely echoed (E1 aligner headroom) / absorbed (unrecoverable by any extractor)
  false-pos   - coincidence-echo null control: invert out_p against a MISMATCHED doc's R;
                every firing is a false positive that would corrupt out_final & inflate
                fact recall
  reader-bias - u_qa on all-placeholder vs tau_walk vs no_privacy doc_p: does the SQuAD
                reader unfairly penalize placeholder-laden text?

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/surrogate_env_diagnostics.py
"""
import json
import re
import time
from pathlib import Path
from statistics import mean

from cloak.corpora import load_task_docs, refs_of
from cloak.extract import invert
from cloak.tasks import TASK_TEMPLATE
from cloak.train.probes import probes_for_docs
from cloak.train.reward import _sentences, u_qa
from inferdpt.llm import LLMClient, _cache_path

import sys
sys.path.append(str(Path(__file__).resolve().parent / "spikes"))  # runner scripts may live there
from surrogate_validation import build_arms  # noqa: E402

GEN_PARAMS = {"temperature": 0.0, "max_tokens": 512,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
GEN_MODEL = "gemma 4 (E4B)"  # = roundtrip.RT_MODEL pin (2026-07-05)
TAU = 0.02
_emb = None


_remote = None


def cached_out_p(corpus: str, doc_p: str) -> str | None:
    """out_p for the artifact's doc_p: disk cache first, remote generate on miss (cached)."""
    global _remote
    msgs = [{"role": "user", "content": TASK_TEMPLATE[corpus].format(doc=doc_p)}]
    p = _cache_path(GEN_MODEL, msgs, GEN_PARAMS)
    if p and Path(p).exists():
        return json.loads(Path(p).read_text())["content"]
    if _remote is None:
        _remote = LLMClient(GEN_MODEL, **GEN_PARAMS)
    return _remote.generate(TASK_TEMPLATE[corpus].format(doc=doc_p))


def _gen_fired(stats: dict) -> int:
    return (stats["gen_exact"] + stats["gen_fuzzy"] + stats.get("gen_semantic", 0) +
            stats.get("gen_pointer", 0))


def _embedder():
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb


def check_absent(corpora: dict) -> dict:
    """Per unique gen replacement in tau_walk/all_floor arms: echo class in cached out_p."""
    from rapidfuzz import fuzz
    out, examples = {}, []
    for corpus, (docs, arms_of) in corpora.items():
        rows = []
        for d in docs:
            for arm in ("tau_walk", "all_floor"):
                doc_p, R = arms_of[d["id"]][arm]
                op = cached_out_p(corpus, doc_p)
                if op is None:
                    continue
                sents = _sentences(op) or [op]
                sent_vecs = _embedder().encode(sents, normalize_embeddings=True)
                seen = set()
                for e in R:
                    if e["action"] != "generalize" or e["replacement"].lower() in seen:
                        continue
                    seen.add(e["replacement"].lower())
                    if re.search(rf"\b{re.escape(e['replacement'])}\b", op, re.IGNORECASE):
                        cls, fz, sim = "exact", 100.0, 1.0
                    else:
                        fz = fuzz.partial_ratio(e["replacement"].lower(), op.lower())
                        v = _embedder().encode([e["replacement"]], normalize_embeddings=True)
                        sim = float((sent_vecs @ v.T).max())
                        cls = ("fuzzy90" if fz >= 90 else
                               "loose" if fz >= 70 or sim >= 0.6 else "absorbed")
                        if cls == "loose" and len(examples) < 12:
                            best = sents[int((sent_vecs @ v.T).argmax())]
                            examples.append({"corpus": corpus, "replacement": e["replacement"],
                                             "surface": e["surface"], "fuzz": round(fz, 1),
                                             "sim": round(sim, 3), "best_sent": best[:120]})
                    rows.append({"arm": arm, "cls": cls, "fuzz": round(fz, 1),
                                 "sim": round(sim, 3)})
        n = len(rows)
        dist = {c: sum(r["cls"] == c for r in rows) for c in
                ("exact", "fuzzy90", "loose", "absorbed")}
        out[corpus] = {"n_replacements": n,
                       "fractions": {c: round(v / n, 3) for c, v in dist.items()} if n else {}}
        print(f"[absent] {corpus}: n={n} {out[corpus]['fractions']}", flush=True)
    out["loose_examples"] = examples
    return out


def check_false_pos(corpora: dict) -> dict:
    """Invert each tau_walk out_p against the NEXT doc's R (null control)."""
    out = {}
    for corpus, (docs, arms_of) in corpora.items():
        pairs = [(d, cached_out_p(corpus, arms_of[d["id"]]["tau_walk"][0])) for d in docs]
        pairs = [(d, op) for d, op in pairs if op]
        fired_null, fired_match, n_null, n_match = 0, 0, 0, 0
        for i, (d, op) in enumerate(pairs):
            R_own = arms_of[d["id"]]["tau_walk"][1]
            R_other = arms_of[pairs[(i + 1) % len(pairs)][0]["id"]]["tau_walk"][1]
            gens_own = {e["replacement"].lower() for e in R_own if e["action"] == "generalize"}
            gens_other = {e["replacement"].lower() for e in R_other if e["action"] == "generalize"}
            _, st_m = invert(op, R_own)
            _, st_n = invert(op, R_other)
            fired_match += _gen_fired(st_m)
            fired_null += _gen_fired(st_n)
            n_match += len(gens_own)
            n_null += len(gens_other)
        out[corpus] = {"matched_fire_rate": round(fired_match / n_match, 3) if n_match else None,
                       "null_fire_rate": round(fired_null / n_null, 3) if n_null else None,
                       "n_matched": n_match, "n_null": n_null}
        print(f"[false-pos] {corpus}: {out[corpus]}", flush=True)
    return out


def placeholder_arm(text: str, R_tau: list[dict]) -> tuple[str, list[dict]]:
    """Every substituted span (from the artifact's tau_walk R, which carries offsets)
    -> typed indexed placeholder (one index per unique surface)."""
    counters, by_surface, R, out = {}, {}, [], text
    for e in sorted(R_tau, key=lambda e: -e["start"]):
        ph = by_surface.get(e["surface"].lower())
        if ph is None:
            counters[e["type"]] = counters.get(e["type"], 0) + 1
            ph = f"<{e['type']}_{counters[e['type']]}>"
            by_surface[e["surface"].lower()] = ph
        R.append({"surface": e["surface"], "type": e["type"], "action": "placeholder",
                  "replacement": ph})
        out = out[:e["start"]] + ph + out[e["end"]:]
    return out, R[::-1]


def check_reader_bias(corpora: dict) -> dict:
    """u_qa (doc_p side) on no_privacy / tau_walk / all_placeholder for the same docs."""
    out = {}
    for corpus, (docs, arms_of) in corpora.items():
        probes = probes_for_docs(docs, {d["id"]: arms_of[d["id"]]["tau_walk"][1] for d in docs},
                                 workers=6)
        arm_scores = {"no_privacy": [], "tau_walk": [], "all_placeholder": []}
        examples = []
        for d in docs:
            ps = probes[d["id"]]
            if not ps:
                continue
            variants = {"no_privacy": (d["text"], []),
                        "tau_walk": arms_of[d["id"]]["tau_walk"],
                        "all_placeholder": placeholder_arm(
                            d["text"], arms_of[d["id"]]["tau_walk"][1])}
            for arm, (doc_p, R) in variants.items():
                score, det_rows = u_qa(doc_p, R, ps)
                if score is not None:
                    arm_scores[arm].append(score)
                if arm == "all_placeholder" and det_rows and len(examples) < 8:
                    examples.append({"doc": d["id"], **det_rows[0]})
        out[corpus] = {a: {"u_qa": round(mean(v), 4) if v else None, "n": len(v)}
                       for a, v in arm_scores.items()}
        out[corpus]["placeholder_examples"] = examples
        print(f"[reader-bias] {corpus}: " +
              " ".join(f"{a}={out[corpus][a]['u_qa']}" for a in arm_scores), flush=True)
    return out


def main():
    t0 = time.time()
    from build_arms_artifact import ARTIFACT, load_artifact
    art = load_artifact()  # build_arms_artifact.py must have run (detection is
    print(f"arms artifact {ARTIFACT}", flush=True)  # cross-process nondeterministic)
    corpora = {}
    for corpus in ("clinical", "enron", "aeslc"):
        docs = load_task_docs(corpus, 16)
        corpora[corpus] = (docs, {i: {a: (dp, R) for a, (dp, R) in arms.items()}
                                  for i, arms in art[corpus].items()})

    report = {"tau": TAU, "gen_model": GEN_MODEL, "docs_per_corpus": 16,
              "absent_decomposition": check_absent(corpora),
              "inversion_false_positives": check_false_pos(corpora)}
    report["reader_placeholder_bias"] = check_reader_bias(corpora)

    out = Path("results/surrogate_env_diagnostics.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
