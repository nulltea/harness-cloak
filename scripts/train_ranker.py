"""Stage-1 ranker training: contextual bandit, REINFORCE + KL leash, fully local reward.

Implements spec §2 Phase 1 (docs/specs/RL/surrogate-ranker-infiller.md) on the Phase-0
environment (data/ranker_env.json + the arms artifact):

  per doc, per step: sample G level-assignments inside the count-floor plumbing
  (default k_floors are inert at 1.0; walk_risk is offline-only) ->
  assemble doc_p/R (injectivity via a DYNAMIC sampling mask: claimed fills unsampleable) ->
  r = alpha*(1 - A) + (1-alpha)*u_qa(train-split probes)
  (A = mean fill_proximity over level-mode fills; the action table's cached "p6" IS
  fill_proximity = cos_MiniLM(fill, original) — identical numbers, deterministic) ->
  group-relative advantage -> REINFORCE update of the feature policy + kl_coef*KL(pi || pi_0).

pi_0 = behavior clone of the floor-walk (min-aset legal level) — never RL from random; under
--randomize-floors both BC and RL sample per-episode floors (spec §5.4). Known limitation
(accepted for the stage-1 ablation): BC and the KL reference are computed over the STATIC
floor-legal sets while rollouts sample under the dynamic injectivity mask — a small
policy/reference mismatch on the spans whose fills collide under the BC trajectory. Policy =
cloak.train.ranker.RankerPolicy (feature-only; the plan's ablation floor promoted to v0).
Placeholder tokens are assigned per rollout at assemble time; direct identifiers keep the
artifact's chain tokens. The echo channel is deliberately unpriced (spec §5.2).

Outputs per alpha: data/ranker_policy_a{alpha}.pt + results/ranker_train_a{alpha}.json
(per-epoch mean reward/A/U, placeholder rate, KL).

Run (full, ~10-20 min/alpha):  PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_ranker.py
Smoke (~2 min):                ... scripts/train_ranker.py --smoke
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from cloak.train.ranker import (EncoderPolicy, RankerPolicy, action_features,
                                span_context)
from cloak.train.reward import canon, fact_f1s, stage1_reward, u_qa
from cloak.train.qa_builder import (AciTaskAdapter,
                                    action_is_floor_legal,
                                    builder_pin,
                                    coarsest_entailing_legal_action,
                                    context_reader_pin,
                                    effective_count_floors,
                                    floor_for_runtime_type,
                                    frozen_occurrences_from_arms,
                                    freeze_ranker_environment,
                                    normalize_cost_budgets,
                                    normalize_family_budgets,
                                    normalize_threshold_manifest,
                                    relation_teacher_pin,
                                    utility_assertion_semantic_key,
                                    utility_scorer_pin)
from cloak.train.utility_credit import provisional_advantages
from cloak.tasks import SCHEMA_CORPORA
from cloak.runtime_types import PLACEHOLDER_RE, placeholder_token, placeholder_type_token

try:  # surrogate-only environments run without the round-trip module
    from cloak.train.roundtrip import roundtrip_batch
except ImportError:
    roundtrip_batch = None


def _ctx_of(doc, i):
    """Span i's precomputed context embedding, or None (MLP mode has no doc['ctx']).
    set_context(None) is a no-op on RankerPolicy; in encoder mode doc['ctx'] is always set."""
    ctx = doc.get("ctx")
    return None if ctx is None else ctx[i]


_DEFAULT_DECISIONS = object()


def _roundtrip_job(doc, doc_p, R, *, decisions=_DEFAULT_DECISIONS):
    """Build a roundtrip_batch job, preserving the legacy schema unless carrier fields exist."""
    job = {"doc_id": doc["id"], "corpus": doc["corpus"], "doc_p": doc_p, "R": R,
           "probes": doc["probes_train"]}
    if "utility_artifact" in doc:
        job["utility_artifact"] = doc["utility_artifact"]
    for key in ("ladder", "decisions", "out_hi", "schema"):
        if key == "decisions" and decisions is not _DEFAULT_DECISIONS:
            job[key] = decisions
        elif key in doc:
            job[key] = doc[key]
    return job


def _span_credit_decisions(doc):
    return [entry for entry in doc.get("decisions", []) if entry.get("span_ids")]


def _counterfactual_roundtrip_job(doc, doc_p, R):
    if "decisions" not in doc:
        return _roundtrip_job(doc, doc_p, R)
    return _roundtrip_job(doc, doc_p, R, decisions=_span_credit_decisions(doc))


def _artifact_docs(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    return payload.get("docs", payload) if isinstance(payload, dict) else {}


def _artifact_entries(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict) and e.get("kept", True) is not False]
    if isinstance(payload, dict):
        entries = (payload.get("entries") or payload.get("train") or payload.get("probes")
                   or payload.get("ladder") or payload.get("decisions") or [])
        return [e for e in entries if isinstance(e, dict) and e.get("kept", True) is not False]
    return []


def _artifact_out_hi(*payloads):
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("out_hi"):
            return payload["out_hi"]
    return None


def attach_utility_artifact(docs, artifact):
    """Attach one frozen v2 artifact without applying legacy probe-count filtering."""
    artifact_docs = artifact.get("documents", {})
    attached = []
    for doc in docs:
        state = artifact_docs.get(doc["id"])
        if not state:
            raise ValueError(f"utility artifact is missing document state for {doc['id']!r}")
        if state.get("measurement_state") in {"unsupported", "build_failed"}:
            raise ValueError(
                f"utility artifact document {doc['id']!r} has unusable measurement_state "
                f"{state['measurement_state']!r}"
            )
        assertion_ids = state.get("assertion_ids") or []
        if not assertion_ids:
            raise ValueError(f"utility artifact document {doc['id']!r} has no accepted assertions")
        next_doc = dict(doc)
        next_doc["utility_artifact"] = artifact
        attached.append(next_doc)
    return attached


# ---------- assembly (rollout -> doc_p, R) ----------

def _case_adjust(fill: str, text: str, start: int) -> str:
    """substitute.py's sentence-start casing, applied at the decision occurrence."""
    prev = text[:start].rstrip()
    sent_start = not prev or prev[-1] in ".!?\n"
    return (fill[0].upper() if sent_start else fill[0].lower()) + fill[1:]


_CLEANUP_DELETIONS = (
    re.compile(r"\b([Aa]n?|[Tt]he) (?=(?:an?|the)\b)"),  # duplicate article
    re.compile(r"\b[Ii]n (?=in\b)"),                     # 'in in'
)


def _cleanup_tracked(out: str, spans: list[dict]) -> tuple[str, int]:
    """_cleanup as tracked deletions over recorded spans (each span: {"box": [s0, s1]}).

    Each cleanup regex is a pure deletion. For every match [d0, d1): shift boxes lying
    fully to its right left by the deletion length; boxes fully left are untouched; a box
    whose interior the deletion overlaps is dropped (box -> None) and counted. Matches are
    applied right-to-left so their (d0, d1) stay valid in the shrinking string; the untouched
    left prefix means each match's original offsets equal its current-string offsets, so the
    match and box comparisons live in one coordinate system."""
    dropped = 0
    for pat in _CLEANUP_DELETIONS:
        matches = [(m.start(), m.end()) for m in pat.finditer(out)]
        for d0, d1 in sorted(matches, reverse=True):
            out = out[:d0] + out[d1:]
            length = d1 - d0
            for sp in spans:
                box = sp["box"]
                if box is None:
                    continue
                if d1 <= box[0]:        # deletion fully left of span -> shift span left
                    box[0] -= length
                    box[1] -= length
                elif d0 >= box[1]:      # deletion fully right of span -> unchanged
                    continue
                else:                   # deletion overlaps span interior -> drop it
                    sp["box"] = None
                    dropped += 1
    return out, dropped


def assemble(text: str, R_walk: list[dict], spans: list[dict],
             choice: dict[str, dict]) -> tuple[str, list[dict]]:
    """doc_p and rollout R from per-surface choices; exactly reproduces the deployed
    substitute() surface forms (casing at the decision occurrence, article cleanup).

    Injectivity is guaranteed UPSTREAM by the dynamic sampling mask (rollout_reward) —
    a collision here is a bug, not an input. Placeholder tokens: reuse the artifact's
    token when the walk also placeholder'd that surface (exact BC reproduction);
    otherwise mint fresh tokens seeded ABOVE the artifact's max index per type, so
    rollout tokens can never collide with the artifact's direct-identifier tokens.
    """
    art_ph = {e["surface"].lower(): e["replacement"] for e in R_walk
              if e["action"] == "placeholder"}
    counters: dict[str, int] = {}
    for e in R_walk:  # seed above existing <TYPE_n> indices
        m = PLACEHOLDER_RE.fullmatch(e["replacement"])
        if m:
            body = e["replacement"][1:-1]
            typ, idx = body.rsplit("_", 1)
            counters[typ] = max(counters.get(typ, 0), int(idx))
    ph_by_surface: dict[str, str] = {}
    used: dict[str, str] = {}
    fills: dict[str, dict] = {}

    def placeholder(skey: str, typ: str) -> str:
        if skey not in ph_by_surface:
            if skey in art_ph:
                ph_by_surface[skey] = art_ph[skey]
            else:
                tok = placeholder_type_token(typ)
                counters[tok] = counters.get(tok, 0) + 1
                ph_by_surface[skey] = placeholder_token(typ, counters[tok])
        return ph_by_surface[skey]

    for s in spans:  # decision spans, walk order (deterministic)
        skey = s["surface"].lower()
        c = choice[skey]
        if c["mode"] == "level":
            fill = _case_adjust(c["fill"], text, s["start"])
            assert used.setdefault(fill.lower(), skey) == skey, \
                f"injectivity violated at assemble: {fill!r}"  # masked upstream
            fills[skey] = {"replacement": fill, "action": "generalize"}
        else:
            fills[skey] = {"replacement": placeholder(skey, s["type"]),
                           "action": "placeholder"}

    out, R = text, []
    seen: dict[tuple[str, str], dict] = {}
    # spans: one record per APPLIED replacement, {"box": [start, end], "entry": R-entry};
    # box tracked through the right-to-left pass and _cleanup into FINAL doc_p offsets.
    spans_rec: list[dict] = []
    for e in sorted(R_walk, key=lambda e: -e["start"]):
        skey = e["surface"].lower()
        # apply the decision only to occurrences the walk treated as quasi (they carry a
        # lattice); a same-surface occurrence typed as a DIRECT identifier keeps its chain
        # token — per-occurrence typing wins, exactly as in substitute()
        if skey in fills and e.get("lattice"):
            rep, act = fills[skey]["replacement"], fills[skey]["action"]
        else:
            rep, act = e["replacement"], e["action"]
        start, end = e["start"], e["end"]
        out = out[:start] + rep + out[end:]
        # right-to-left: every already-recorded span lies to the right of this edit, so all
        # shift by the length delta; the new span sits at (start, start+len(rep))
        delta = len(rep) - (end - start)
        for sp in spans_rec:
            sp["box"][0] += delta
            sp["box"][1] += delta
        # R must cover every APPLIED replacement: mixed-typing surfaces legally map one
        # surface to two replacements (e.g. 'participant'→'a person' AND '<PERSON_1>'),
        # and dropping either breaks inversion of out_p
        key = (skey, rep.lower())
        if key not in seen:
            seen[key] = {"surface": e["surface"], "type": e["type"],
                         "action": act, "replacement": rep}
            R.append(seen[key])
        spans_rec.append({"box": [start, start + len(rep)], "entry": seen[key]})

    out, _dropped = _cleanup_tracked(out, spans_rec)

    for entry in R:
        entry["fill_spans"] = []
    for sp in spans_rec:
        if sp["box"] is not None:
            sp["entry"]["fill_spans"].append(sp["box"])
    for entry in R:
        entry["fill_spans"].sort()
        for s0, s1 in entry["fill_spans"]:
            assert out[s0:s1] == entry["replacement"], (
                f"fill_spans invariant: doc_p[{s0}:{s1}]={out[s0:s1]!r} != "
                f"{entry['replacement']!r}")
    return out, R


def derive_spans(raw_spans, floors, corpus, device):
    """Legal set + floor-walk BC teacher + features from per-type count floors.
    legal = placeholder ∪ {levels with aset >= floor[type]} (walk_risk is offline-only now).
    bc_action = the legal NON-KEEP level minimizing (aset, index) — the most specific
    non-original level, by min aset not list order (actions["aset"] is not always sorted);
    placeholder fallback when no non-KEEP level is legal. Every span keeps a placeholder so
    legal is never empty."""
    spans, feats = [], []
    effective_floors = effective_count_floors({"k_floors": floors})
    for s in raw_spans:
        s = dict(s)
        # unknown span types inherit the OTHER floor (default-deny) — never a silent waiver
        k = floor_for_runtime_type(s["type"], effective_floors)
        s["legal"] = [i for i, a in enumerate(s["actions"])
                      if action_is_floor_legal(a, k)]
        ph_idx = next(i for i, a in enumerate(s["actions"]) if a["mode"] == "placeholder")
        s["bc_action"] = min(((a.get("aset", 0), i) for i, a in enumerate(s["actions"])
                              if a["mode"] == "level" and not a.get("keep")
                              and a.get("aset", 0) >= k),
                             default=(None, ph_idx))[1]
        spans.append(s)
        feats.append(action_features(s, corpus, k).to(device))
    return spans, feats


def floor_walk_choice(spans):
    """THE floor-walk baseline choice with the walk-order collision rule (first-come keeps
    the fill, later colliders fall back to placeholder) — shared by ExIt, the support scan,
    and any baseline consumer, so the gate certifies the same baseline training uses."""
    used, choice = set(), {}
    for s in spans:
        a = s["actions"][s["bc_action"]]
        if a["mode"] == "level" and a["fill"].lower() in used:
            a = s["actions"][next(i for i, x in enumerate(s["actions"])
                                  if x["mode"] == "placeholder")]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
        choice[s["surface"].lower()] = a
    return choice


def verify_bc_reproduction(docs, art) -> int:
    """Invariant: assemble(behavior-clone choices) == the artifact's tau_walk doc_p."""
    bad = 0
    for doc in docs:
        choice = {s["surface"].lower(): s["actions"][s["bc_action"]] for s in doc["spans"]}
        doc_p, _ = assemble(doc["text"], doc["R_walk"], doc["spans"], choice)
        ref = art[doc["corpus"]][doc["id"]]["tau_walk"][0]
        if doc_p != ref:
            bad += 1
            print(f"BC-REPRODUCTION MISMATCH {doc['id']}", flush=True)
    return bad


# ---------- reward ----------

def sample_rollout(doc, span_rows, feats, policy, greedy=False, decision_ids=None):
    """Sampling half of a rollout under the DYNAMIC injectivity mask (spec §3.3-1).
    Returns (choice, logps, ph_rate, doc_p, R, legals) — no reward computed here. `legals`
    is the per-span DYNAMIC legal set actually sampled from (walk order), so entropy/KL can
    be scored over the masks the policy really used, not the static floor-legal sets. When
    stable decision IDs are supplied, `logps` is keyed by those IDs rather than span position."""
    if decision_ids is not None and len(decision_ids) != len(span_rows):
        raise ValueError("sampled decision IDs must align with span rows")
    used: set[str] = set()
    choice, logps, legals, n_level = {}, {} if decision_ids is not None else [], [], 0
    for i, (s, f) in enumerate(zip(span_rows, feats)):
        policy.set_context(_ctx_of(doc, i))
        legal_dyn = [j for j in s["legal"]
                     if s["actions"][j]["mode"] == "placeholder"
                     or s["actions"][j]["fill"].lower() not in used]
        a_idx, lp = policy.sample(f, legal_dyn, greedy=greedy)
        a = s["actions"][a_idx]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
            n_level += 1
        choice[s["surface"].lower()] = a
        if decision_ids is None:
            logps.append(lp)
        else:
            decision_id = str(decision_ids[i])
            if decision_id in logps:
                raise ValueError(f"duplicate sampled decision id {decision_id!r}")
            logps[decision_id] = lp
        legals.append(legal_dyn)
    doc_p, R = assemble(doc["text"], doc["R_walk"], span_rows, choice)
    return choice, logps, 1.0 - n_level / len(span_rows), doc_p, R, legals


def rollout_reward(doc, span_rows, feats, policy, alpha, greedy=False):
    """One rollout with the DYNAMIC injectivity mask (spec §3.3-1: claimed levels are
    unsampleable, not downgraded post-hoc): spans are decided sequentially in walk order;
    a level whose fill is already claimed by a different surface is masked out before
    sampling, so log-probs, A, and ph_rate all describe the action actually executed."""
    choice, logps, ph_rate, doc_p, R, _ = sample_rollout(doc, span_rows, feats, policy, greedy)
    p6s = [c["p6"] for c in choice.values() if c["mode"] == "level"]
    A = sum(p6s) / len(p6s) if p6s else 0.0
    U, _ = u_qa(doc_p, R, doc["probes_train"])
    r = stage1_reward(A, U, alpha)
    return r, {"A": A, "U": U or 0.0, "ph_rate": ph_rate}, logps


def rloo_advantage(rt: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline, NO std normalization (Dr.GRPO correction; spec Phase 2)."""
    G = rt.numel()
    return (rt - rt.mean()) * G / (G - 1)


def _decision_key_mapping(rows, controlled_ids, *, doc_id, error_type):
    mapping = {}
    seen_ids = set()
    for row in rows:
        decision_id = str(row.get("decision_id"))
        runtime_type = row.get("runtime_type")
        canonical_key = row.get("canonical_key")
        if (decision_id not in controlled_ids or runtime_type is None
                or canonical_key is None or decision_id in seen_ids):
            raise error_type(f"document {doc_id!r} has invalid frozen decision keys")
        key = (str(runtime_type), str(canonical_key))
        if key in mapping:
            raise error_type(f"document {doc_id!r} has duplicate frozen decision keys")
        mapping[key] = decision_id
        seen_ids.add(decision_id)
    if set(mapping.values()) != controlled_ids:
        raise error_type(f"document {doc_id!r} decision keys do not cover controlled decisions")
    return mapping


def utility_decision_ids(doc, span_rows):
    """Bind sampled span rows to frozen stable decision IDs without positional semantics."""
    state = doc["utility_artifact"]["documents"][doc["id"]]
    controlled_ids = {str(value) for value in state["controlled_decision_ids"]}
    decision_by_key = _decision_key_mapping(
        state.get("decision_keys", []),
        controlled_ids,
        doc_id=doc["id"],
        error_type=ValueError,
    )

    sampled_ids = []
    for span in span_rows:
        key = (str(span["type"]), canon(str(span["surface"])))
        decision_id = decision_by_key.get(key)
        if decision_id is None:
            raise ValueError(f"document {doc['id']!r} has no frozen decision for span {key}")
        sampled_ids.append(decision_id)
    if len(sampled_ids) != len(set(sampled_ids)) or set(sampled_ids) != controlled_ids:
        raise ValueError(f"document {doc['id']!r} sampled spans do not match controlled decisions")
    return sampled_ids


def structured_utility_loss(doc_id, log_probs, provisional, *, counterfactual_losses=None):
    """Apply one v2 utility term per rollout-decision pair at fixed 1/G weight."""
    counterfactual_losses = counterfactual_losses or {}
    terms = []
    for rollout_index, per_decision in enumerate(log_probs):
        for decision_id, log_prob in per_decision.items():
            counterfactual = counterfactual_losses.get((doc_id, rollout_index, decision_id))
            if counterfactual is not None:
                terms.append(counterfactual)
            else:
                terms.append(-provisional[(rollout_index, decision_id)] * log_prob)
    if not terms:
        raise ValueError(f"document {doc_id!r} has no policy decisions")
    return sum(terms) / len(log_probs)


def policy_entropy(policy, feats, legal) -> torch.Tensor:
    lp = policy.log_probs(feats, legal)
    return -(lp.exp() * lp).sum()


def counterfactual_terms(doc, policy, choice, logps, base_r=None, *, frac, rng, rt_workers):
    """Exact per-span credit (spec Phase 2; COMA made exact by reward determinism):
    for a sampled fraction of non-placeholder spans, re-run the round trip with ONLY that
    span flipped to its placeholder; adv_s = base_r - r_cf weights that span's logp.
    Counterfactual doc_p's are cache-friendly (identical across epochs at fixed choices).
    base_r is accepted for old call sites; the restricted base reward is recomputed here."""
    cand = [i for i, s in enumerate(doc["spans"])
            if choice[s["surface"].lower()]["mode"] == "level"]
    take = [i for i in cand if rng.random() < frac]
    if not take:
        return 0.0, 0
    base_doc_p, base_R = assemble(doc["text"], doc["R_walk"], doc["spans"], choice)
    jobs = [_counterfactual_roundtrip_job(doc, base_doc_p, base_R)]
    for i in take:
        s = doc["spans"][i]
        cf = dict(choice)
        ph_idx = next(k for k, a in enumerate(s["actions"]) if a["mode"] == "placeholder")
        cf[s["surface"].lower()] = s["actions"][ph_idx]
        doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], cf)
        jobs.append(_counterfactual_roundtrip_job(doc, doc_p, R))
    res = roundtrip_batch(jobs, workers=rt_workers)
    base_r = res[0]["recall"] or 0.0
    term = 0.0
    for i, r in zip(take, res[1:]):
        adv_s = base_r - (r["recall"] or 0.0)
        term = term - adv_s * logps[i]
    return term, len(take)


def train_roundtrip(docs, policy, *, G, epochs, lr, entropy_coef, kl_coef, ref,
                    rt_workers, seed, cf_frac=0.0, log_rows=None,
                    counterfactual_losses=None, utility_reward_cache=None):
    """RLOO + tie-filter epoch loop against roundtrip_batch. Returns per-epoch stat rows.
    Artifact-backed documents use per-decision v2 utility credit. Legacy scalar documents retain
    their existing RLOO and optional separate legacy counterfactual path."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rows = []
    for epoch in range(epochs):
        rng = random.Random(seed * 1000 + epoch)
        order = list(range(len(docs)))
        rng.shuffle(order)
        ep = {
            "r": [], "ph": [], "ent": [], "ties_skipped": 0, "cf_used": 0,
            "qa_cache_hits": 0, "qa_cache_misses": 0,
        }
        for di in order:
            doc = docs[di]
            logps_l, ph_l, legals_l = [], [], []
            jobs = []
            cache_inputs = []
            decision_ids = (utility_decision_ids(doc, doc["spans"])
                            if "utility_artifact" in doc else None)
            scorer_pin = (
                _utility_scorer_pin(doc)
                if decision_ids is not None and utility_reward_cache is not None else None
            )
            for _ in range(G):
                choice, logps, ph, doc_p, R, legals = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy, decision_ids=decision_ids)
                jobs.append(_roundtrip_job(doc, doc_p, R))
                if decision_ids is not None and utility_reward_cache is not None:
                    cache_inputs.append({
                        "doc_id": doc["id"],
                        "action_vector": utility_action_vector(
                            doc["spans"], decision_ids, choice
                        ),
                        "doc_p": doc_p,
                        "artifact_hash": doc["utility_artifact"]["artifact_hash"],
                        "scorer_pin": scorer_pin,
                    })
                logps_l.append(logps)
                ph_l.append(ph)
                legals_l.append(legals)
            if cache_inputs:
                hits_before = utility_reward_cache.hits
                misses_before = utility_reward_cache.misses
                res = cached_utility_roundtrips(
                    jobs, cache_inputs, utility_reward_cache, workers=rt_workers
                )
                ep["qa_cache_hits"] += utility_reward_cache.hits - hits_before
                ep["qa_cache_misses"] += utility_reward_cache.misses - misses_before
            else:
                res = roundtrip_batch(jobs, workers=rt_workers)
            rt = torch.tensor([r["recall"] or 0.0 for r in res])
            ep["r"].append(rt.mean().item())
            ep["ph"].append(sum(ph_l) / G)
            if "utility_artifact" in doc:
                vectors = [row.get("component_scores") for row in res]
                if not all(isinstance(vector, Mapping) for vector in vectors):
                    raise ValueError(
                        f"document {doc['id']!r} has a utility artifact but no component scores"
                    )
                state = doc["utility_artifact"]["documents"][doc["id"]]
                provisional = provisional_advantages(
                    vectors,
                    doc["utility_artifact"],
                    state["occurrence_to_decision"],
                    doc_id=doc["id"],
                )
                pg = structured_utility_loss(
                    doc["id"],
                    logps_l,
                    provisional,
                    counterfactual_losses=counterfactual_losses,
                )
            else:
                if rt.max() == rt.min():                  # DAPO tie filter
                    ep["ties_skipped"] += 1
                    continue
                adv = rloo_advantage(rt)
                pg = -sum(a * torch.stack(lp).sum() for a, lp in zip(adv, logps_l)) / G
            # entropy over the DYNAMIC masks each rollout actually sampled from (not the
            # static floor-legal sets), mean over spans and rollouts
            ent, n_ent = 0.0, 0
            for legals in legals_l:
                for i, (s, f) in enumerate(zip(doc["spans"], doc["feats"])):
                    policy.set_context(_ctx_of(doc, i))
                    ent = ent + policy_entropy(policy, f, legals[i])
                    n_ent += 1
            ent = ent / max(n_ent, 1)
            loss = pg - entropy_coef * ent
            if kl_coef > 0 and ref is not None:
                # KL over each rollout's recorded dynamic legal set, aligned per rollout,
                # mean over spans and rollouts
                kl, n_kl = 0.0, 0
                for legals in legals_l:
                    for i, (s, f) in enumerate(zip(doc["spans"], doc["feats"])):
                        policy.set_context(_ctx_of(doc, i))
                        ref.set_context(_ctx_of(doc, i))
                        kl = kl + kl_to_ref(policy, ref, f, legals[i])
                        n_kl += 1
                loss = loss + kl_coef * kl / max(n_kl, 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep["ent"].append(ent.item())
            if cf_frac > 0 and "utility_artifact" not in doc:
                g_choice, g_logps, _, _, _, _ = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy, greedy=True)
                term, n_cf = counterfactual_terms(doc, policy, g_choice, g_logps,
                                                  frac=cf_frac, rng=rng, rt_workers=rt_workers)
                if n_cf > 0 and isinstance(term, torch.Tensor):
                    opt.zero_grad()
                    term.backward()
                    opt.step()
                    ep["cf_used"] += n_cf
        n = max(len(ep["r"]), 1)
        row = {"epoch": epoch, "r": round(sum(ep["r"]) / n, 4),
               "ph": round(sum(ep["ph"]) / n, 4),
               "ent": round(sum(ep["ent"]) / max(len(ep["ent"]), 1), 4),
               "ties_skipped": ep["ties_skipped"], "cf_used": ep["cf_used"],
               "qa_cache_hits": ep["qa_cache_hits"],
               "qa_cache_misses": ep["qa_cache_misses"]}
        rows.append(row)
        if utility_reward_cache is not None:
            print(
                f"[roundtrip] QA reward cache hits={ep['qa_cache_hits']} "
                f"misses={ep['qa_cache_misses']}",
                flush=True,
            )
        if log_rows is not None:
            log_rows.append(row)
        print(f"[rt] epoch {epoch}: " +
              " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"), flush=True)
    return rows


# ---------- expert iteration (ExIt) outer loop ----------

def exit_round(docs, policy, *, G, rt_workers, seed, utility_reward_cache=None):
    """One expert-iteration round (spec Phase 2 workhorse): per doc sample G rollouts,
    keep the best strictly beating the floor-walk baseline. Baselines and rollouts all go
    through the round trip. Artifact-backed initial rollouts reuse the QA reward cache;
    refreshed serial verification remains uncached. Returns (winners, stats)."""
    torch.manual_seed(seed)
    jobs, meta = [], []          # baseline job per doc first, then G rollouts per doc
    cache_inputs = []
    bc_jobs = {}
    for di, doc in enumerate(docs):
        # ExIt reference = THE floor-walk baseline via floor_walk_choice (walk-order collision
        # rule resolves colliding fills to placeholder), per spec Phase 2: a rollout is a
        # winner only if it strictly beats the floor-walk round-trip reward. Injective by
        # construction, so assemble() can no longer collide.
        bc_choice = floor_walk_choice(doc["spans"])
        doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], bc_choice)
        job = _roundtrip_job(doc, doc_p, R)
        jobs.append(job)
        bc_jobs[di] = job
        meta.append(("bc", di, None))
        decision_ids = (
            utility_decision_ids(doc, doc["spans"])
            if "utility_artifact" in doc and utility_reward_cache is not None else None
        )
        scorer_pin = (
            _utility_scorer_pin(doc)
            if decision_ids is not None else None
        )
        if decision_ids is not None:
            cache_inputs.append({
                "doc_id": doc["id"],
                "action_vector": utility_action_vector(doc["spans"], decision_ids, bc_choice),
                "doc_p": doc_p,
                "artifact_hash": doc["utility_artifact"]["artifact_hash"],
                "scorer_pin": scorer_pin,
            })
        else:
            cache_inputs.append(None)
        for _ in range(G):
            if decision_ids is None:
                choice, _, _, doc_p, R, _ = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy
                )
            else:
                choice, _, _, doc_p, R, _ = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy, decision_ids=decision_ids
                )
            idx = {s["surface"].lower(): next(
                       i for i, a in enumerate(s["actions"])
                       if a is choice[s["surface"].lower()])
                   for s in doc["spans"]}
            job = _roundtrip_job(doc, doc_p, R)
            jobs.append(job)
            meta.append(("roll", di, idx))
            if decision_ids is not None:
                cache_inputs.append({
                    "doc_id": doc["id"],
                    "action_vector": utility_action_vector(doc["spans"], decision_ids, choice),
                    "doc_p": doc_p,
                    "artifact_hash": doc["utility_artifact"]["artifact_hash"],
                    "scorer_pin": scorer_pin,
                })
            else:
                cache_inputs.append(None)
    cache_hits = cache_misses = 0
    cached_indices = [index for index, inputs in enumerate(cache_inputs) if inputs is not None]
    if cached_indices:
        hits_before = utility_reward_cache.hits
        misses_before = utility_reward_cache.misses
        cached_results = cached_utility_roundtrips(
            [jobs[index] for index in cached_indices],
            [cache_inputs[index] for index in cached_indices],
            utility_reward_cache,
            workers=rt_workers,
        )
        cache_hits = utility_reward_cache.hits - hits_before
        cache_misses = utility_reward_cache.misses - misses_before
        res = [None] * len(jobs)
        for index, result in zip(cached_indices, cached_results):
            res[index] = result
        uncached_indices = [index for index in range(len(jobs)) if index not in cached_indices]
        if uncached_indices:
            uncached_results = roundtrip_batch(
                [jobs[index] for index in uncached_indices], workers=rt_workers
            )
            for index, result in zip(uncached_indices, uncached_results):
                res[index] = result
    else:
        res = roundtrip_batch(jobs, workers=rt_workers)
    it = iter(res)
    bc_r, rolls = {}, {di: [] for di in range(len(docs))}
    for (kind, di, idx), job in zip(meta, jobs):
        r = next(it)["recall"] or 0.0
        if kind == "bc":
            bc_r[di] = r
        else:
            rolls[di].append((r, idx, job))
    winners, clean_bc_r = [], {}
    best_rs = []
    n_candidates = 0
    n_verify_dropped = 0
    for di in range(len(docs)):
        if not rolls[di]:
            continue
        best_r = max(r for r, _, _ in rolls[di])
        best_rs.append(best_r)
        if best_r > bc_r[di]:
            n_candidates += 1
            kept = False
            for r, idx, win_job in rolls[di]:
                if r != best_r:
                    continue
                clean_win = roundtrip_batch(
                    [win_job], workers=1, reader_refresh=True)[0]["recall"] or 0.0
                if di not in clean_bc_r:
                    clean_bc_r[di] = roundtrip_batch(
                        [bc_jobs[di]], workers=1, reader_refresh=True)[0]["recall"] or 0.0
                if clean_win > clean_bc_r[di]:
                    winners.append((di, idx))
                    kept = True
                    break
            if not kept:
                n_verify_dropped += 1
    bc_vals = list(bc_r.values())
    stats = {"mean_best_r": round(sum(best_rs) / max(len(best_rs), 1), 4),
             "mean_bc_r": round(sum(bc_vals) / len(bc_vals), 4) if bc_vals else None,
             "n_candidates": n_candidates,
             "n_verify_dropped": n_verify_dropped,
             "n_winners": len(winners),
             "qa_cache_hits": cache_hits,
             "qa_cache_misses": cache_misses}
    return winners, stats


def clone_choices(policy, items, epochs, lr):
    """SFT on winner action indices — behavior_clone generalized to arbitrary teachers.
    items = (spans, feats, choice_idx) or (spans, feats, choice_idx, ctx) — ctx (encoder
    mode) is the per-span context-embedding list, None/absent for the MLP policy."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        for item in items:
            spans, feats, choice_idx = item[0], item[1], item[2]
            ctx = item[3] if len(item) > 3 else None
            loss = 0.0
            for i, (s, f) in enumerate(zip(spans, feats)):
                policy.set_context(None if ctx is None else ctx[i])
                a_idx = choice_idx[s["surface"].lower()]
                if a_idx not in s["legal"]:
                    continue
                lp = policy.log_probs(f, s["legal"])
                loss = loss - lp[s["legal"].index(a_idx)]
            if isinstance(loss, torch.Tensor):
                opt.zero_grad()
                loss.backward()
                opt.step()
    return policy


# ---------- training ----------

def sample_floors(floors, rng):
    """Per-episode log-uniform floor per type, CENTERED on the deployment default:
    k_T ~ exp(U(ln(max(k/10, 1)), ln(10*k))) — median = k, supported config range [k/10, 10k],
    clamped at 1 from below. This is the supported per-type config range; floors outside
    [k/10, 10k] are extrapolation — the mask still enforces them safely, choice quality is
    untested. Waived types (k <= 1) are NOT randomized: a waiver is a discrete user contract,
    and sampling k > 1 would make keep-original illegal in most episodes exactly where the
    user legalized it. Shared by the RL loop and the floor-randomized BC pretrain."""
    return {t: (1.0 if k <= 1.0 else
                math.exp(rng.uniform(math.log(max(k / 10.0, 1.0)), math.log(10.0 * k))))
            for t, k in floors.items()}


def behavior_clone(policy, docs, epochs, lr, device, floors=None, randomize=False, seed=0):
    """Clone the floor-walk teacher's per-span decisions.

    Fixed floors (randomize off): clone the precomputed spans/feats at the env floors.
    Randomized (randomize on): resample per-type floors per (epoch, doc) with sample_floors
    and clone the teacher derived at those floors, so the KL reference is trained along the
    floor-feature dimension the RL loop queries it on. Seeded from `seed` for reproducibility."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs):
        rng = random.Random(seed * 1000 + epoch) if randomize else None
        for doc in docs:
            if randomize:
                spans, feats = derive_spans(doc["raw_spans"], sample_floors(floors, rng),
                                            doc["corpus"], device)
            else:
                spans, feats = doc["spans"], doc["feats"]
            loss = 0.0
            for i, (s, f) in enumerate(zip(spans, feats)):
                policy.set_context(_ctx_of(doc, i))
                lp = policy.log_probs(f, s["legal"])
                loss = loss - lp[s["legal"].index(s["bc_action"])]
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


def kl_to_ref(policy, ref, feats, legal):
    lp = policy.log_probs(feats, legal)
    with torch.no_grad():
        lq = ref.log_probs(feats, legal)
    return (lp.exp() * (lp - lq)).sum()


def enforce_support_gate(force_ungated: bool, probes_path: str, env_path: str):
    """Round-trip RL training is gated on the support scan
    (results/roundtrip_support_scan.json): verdict must be PASS AND its provenance meta must
    match the live run (rt_model, rt_base_url, probes_path, env_path) — a scan against a
    different endpoint/model/probe set/environment does not certify this run. A missing file
    or missing/stale meta counts as not-passed. --force-ungated bypasses with a loud warning."""
    from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL
    gate = Path("results/roundtrip_support_scan.json")
    reason = None
    if not gate.exists():
        reason = "verdict=MISSING (no scan artifact)"
    else:
        art = json.loads(gate.read_text())
        meta = art.get("meta")
        if art.get("verdict") != "PASS":
            reason = f"verdict={art.get('verdict')}"
        elif not meta:
            reason = "scan artifact has no provenance meta (stale scan; re-run the support scan)"
        else:
            for field, live in (("rt_model", RT_MODEL), ("rt_base_url", RT_BASE_URL),
                                ("probes_path", probes_path), ("env_path", env_path)):
                if meta.get(field) != live:
                    reason = (f"stale scan meta[{field!r}]={meta.get(field)!r} != live {live!r}"
                              "; re-run the support scan for this configuration")
                    break
    if reason is None:
        return
    if force_ungated:
        print(f"WARNING: --force-ungated set — bypassing the round-trip support gate "
              f"({reason}, {gate}). Training on an UNCERTIFIED environment; results "
              "are not gate-backed.", flush=True)
        return
    raise SystemExit(
        f"round-trip support gate not passed ({reason}, {gate}); re-run "
        "scripts/spikes/roundtrip_support_scan.py until it PASSes (or --force-ungated)")


_UTILITY_FLOAT_TOLERANCE = 1e-12


def _utility_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def utility_rollout_cache_identity(
    *,
    doc_id,
    action_vector,
    doc_p,
    out_final=None,
    artifact_hash,
    scorer_pin,
    result_hash=None,
):
    """Return the cache identity for one complete QA utility measurement."""
    identity = {
        "doc_id": str(doc_id),
        "action_vector": {
            str(decision_id): str(action_id)
            for decision_id, action_id in action_vector.items()
        },
        "doc_p": str(doc_p),
        "artifact_hash": str(artifact_hash),
        "scorer_pin": scorer_pin,
    }
    if result_hash is not None:
        identity["result_hash"] = str(result_hash)
    else:
        identity["out_final"] = None if out_final is None else str(out_final)
    return _utility_hash(identity)


class UtilityRewardCache:
    """Append-only content-addressed JSONL cache for complete QA reward results."""

    _VERSION = 3
    _RESULT_VERSION = "utility-roundtrip-result-v1"
    _RESULT_STATUS = "complete"

    def __init__(self, path):
        self.path = Path(path)
        self.entries = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self):
        if not self.path.exists():
            return {}
        try:
            payload = self.path.read_bytes()
            if payload and not payload.endswith(b"\n"):
                raise ValueError("truncated JSONL record")
            entries = {}
            for line_number, encoded_line in enumerate(payload.splitlines(), 1):
                if not encoded_line:
                    raise ValueError(f"malformed JSONL record at line {line_number}")
                record = json.loads(encoded_line.decode("utf-8"))
                if (
                    not isinstance(record, Mapping)
                    or record.get("version") != self._VERSION
                    or not isinstance(record.get("entry"), Mapping)
                ):
                    raise ValueError(f"invalid cache schema at line {line_number}")
                entry = record["entry"]
                request_identity = str(entry.get("request_identity", ""))
                validated = self._validate_entry(request_identity, entry)
                previous = entries.get(request_identity)
                if previous is not None and previous != validated:
                    raise ValueError(
                        f"conflicting duplicate request identity at line {line_number}"
                    )
                entries.setdefault(request_identity, validated)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid utility reward cache {self.path}: {error}") from error
        return entries

    @staticmethod
    def _request_identity(**inputs):
        return utility_rollout_cache_identity(**inputs, out_final=None)

    def request_identity(self, **inputs):
        return self._request_identity(**inputs)

    def lookup(self, **inputs):
        request_identity = self._request_identity(**inputs)
        entry = self.entries.get(request_identity)
        if entry is None:
            self.misses += 1
            return None
        entry = self._validate_entry(request_identity, entry)
        self.hits += 1
        return json.loads(json.dumps(entry["result"], sort_keys=True))

    def store(self, *, result, **inputs):
        return self.store_many([(inputs, result)])[0]

    def store_many(self, items):
        """Validate a dispatched miss batch, then append only its new identities."""
        staged = {}
        stored_results = []
        for inputs, result in items:
            request_identity = self._request_identity(**inputs)
            stored_result = self._canonical_result(result)
            result_hash = _utility_hash(stored_result)
            entry = {
                "request_identity": request_identity,
                "result_hash": result_hash,
                "storage_identity": _utility_hash({
                    "request_identity": request_identity,
                    "result_hash": result_hash,
                }),
                "result": stored_result,
            }
            validated = self._validate_entry(request_identity, entry)
            previous = staged.get(request_identity, self.entries.get(request_identity))
            if previous is not None and previous != validated:
                raise ValueError("conflicting duplicate utility reward cache identity")
            if request_identity not in self.entries:
                staged.setdefault(request_identity, validated)
            stored_results.append(json.loads(json.dumps(stored_result, sort_keys=True)))
        if staged:
            self._persist(staged)
            self.entries.update(staged)
        return stored_results

    @classmethod
    def _canonical_result(cls, result):
        if not isinstance(result, Mapping):
            raise ValueError("utility reward cache requires a complete round-trip result")
        stored = dict(result)
        stored.setdefault("result_version", cls._RESULT_VERSION)
        stored.setdefault("status", cls._RESULT_STATUS)
        required = {
            "result_version", "status", "out_p", "out_final", "component_scores", "recall",
        }
        if not required <= set(stored):
            raise ValueError("utility reward cache requires a complete round-trip result")
        if (
            stored["result_version"] != cls._RESULT_VERSION
            or stored["status"] != cls._RESULT_STATUS
            or not isinstance(stored["out_p"], str)
            or not isinstance(stored["out_final"], str)
            or not isinstance(stored["component_scores"], Mapping)
            or not stored["component_scores"]
        ):
            raise ValueError("utility reward cache requires a complete round-trip result")
        if not cls._valid_score(stored["recall"]):
            raise ValueError("utility reward cache requires a complete round-trip result")
        for assertion_id, score in stored["component_scores"].items():
            if not isinstance(assertion_id, str) or not assertion_id or not cls._valid_score(score):
                raise ValueError("utility reward cache requires a complete round-trip result")
        return json.loads(json.dumps(stored, sort_keys=True, allow_nan=False))

    @staticmethod
    def _valid_score(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )

    @classmethod
    def _validate_entry(cls, request_identity, entry):
        if not isinstance(entry, Mapping):
            raise ValueError("invalid cache entry")
        if entry.get("request_identity") != request_identity:
            raise ValueError("cache request identity mismatch")
        result = cls._canonical_result(entry.get("result"))
        result_hash = _utility_hash(result)
        if entry.get("result_hash") != result_hash:
            raise ValueError("cache result hash mismatch")
        expected_storage_identity = _utility_hash({
            "request_identity": request_identity,
            "result_hash": result_hash,
        })
        if entry.get("storage_identity") != expected_storage_identity:
            raise ValueError("cache storage identity mismatch")
        return {
            "request_identity": request_identity,
            "result_hash": result_hash,
            "storage_identity": expected_storage_identity,
            "result": result,
        }

    def _persist(self, entries):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                {"version": self._VERSION, "entry": entry},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
            for entry in entries.values()
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def utility_action_vector(span_rows, decision_ids, choice):
    """Return a complete stable decision/action vector for one sampled artifact rollout."""
    if len(span_rows) != len(decision_ids):
        raise ValueError("sampled decision IDs must align with action rows")
    vector = {}
    for span, decision_id in zip(span_rows, decision_ids):
        surface = str(span["surface"]).lower()
        if surface not in choice or decision_id in vector:
            raise ValueError("sampled choices must cover each frozen decision exactly once")
        action = dict(choice[surface])
        action.pop("action_id", None)
        action.pop("legal", None)
        action.pop("entails", None)
        action["mode"] = (
            "keep" if action.get("keep") else
            "placeholder" if action.get("mode") == "placeholder" else "level"
        )
        vector[str(decision_id)] = _utility_hash(action)
    return vector


def _utility_scorer_pin(doc):
    from cloak.train.roundtrip import roundtrip_reward_pin

    return roundtrip_reward_pin(
        doc["utility_artifact"]["scorer_pin"],
        corpus=doc["corpus"],
        schema=bool(doc.get("template") == "schema" or doc.get("schema")),
    )


def cached_utility_roundtrips(jobs, cache_inputs, cache, *, workers):
    """Reuse validated artifact-backed base rollouts and persist only complete misses."""
    if len(jobs) != len(cache_inputs):
        raise ValueError("utility cache jobs must align with cache identities")
    results = [None] * len(jobs)
    pending = {}
    miss_inputs, misses = [], []
    for index, inputs in enumerate(cache_inputs):
        request_identity = cache.request_identity(**inputs)
        if request_identity in pending:
            pending[request_identity]["indices"].append(index)
            cache.hits += 1
            continue
        cached = cache.lookup(**inputs)
        if cached is None:
            pending[request_identity] = {"indices": [index], "inputs": inputs}
            miss_inputs.append(inputs)
            misses.append(jobs[index])
        else:
            results[index] = cached
    if misses:
        dispatched = roundtrip_batch(misses, workers=workers)
        if len(dispatched) != len(misses):
            raise ValueError("round-trip result batch does not match dispatched misses")
        stored_results = cache.store_many(list(zip(miss_inputs, dispatched)))
        for inputs, result in zip(miss_inputs, stored_results):
            request_identity = cache.request_identity(**inputs)
            for index in pending[request_identity]["indices"]:
                results[index] = result
    return results


def greedy_roundtrip_readout(docs, policy, *, rt_workers, utility_reward_cache=None):
    """Score the final greedy artifact readout through the complete reward cache."""
    jobs, phs, cache_inputs = [], [], []
    use_cache = utility_reward_cache is not None and all(
        "utility_artifact" in doc for doc in docs
    )
    with torch.no_grad():
        for doc in docs:
            decision_ids = utility_decision_ids(doc, doc["spans"]) if use_cache else None
            choice, _, ph, doc_p, replacements, _ = sample_rollout(
                doc,
                doc["spans"],
                doc["feats"],
                policy,
                greedy=True,
                decision_ids=decision_ids,
            )
            jobs.append(_roundtrip_job(doc, doc_p, replacements))
            phs.append(ph)
            if use_cache:
                cache_inputs.append({
                    "doc_id": doc["id"],
                    "action_vector": utility_action_vector(
                        doc["spans"], decision_ids, choice
                    ),
                    "doc_p": doc_p,
                    "artifact_hash": doc["utility_artifact"]["artifact_hash"],
                    "scorer_pin": _utility_scorer_pin(doc),
                })
    if not use_cache:
        return roundtrip_batch(jobs, workers=rt_workers), phs, {
            "qa_cache_hits": 0,
            "qa_cache_misses": 0,
        }
    hits_before = utility_reward_cache.hits
    misses_before = utility_reward_cache.misses
    results = cached_utility_roundtrips(
        jobs, cache_inputs, utility_reward_cache, workers=rt_workers
    )
    return results, phs, {
        "qa_cache_hits": utility_reward_cache.hits - hits_before,
        "qa_cache_misses": utility_reward_cache.misses - misses_before,
    }


def frozen_training_environment(environment, arms, docs, *, floors=None):
    """Freeze exactly the documents loaded for one artifact-backed training invocation."""
    selected_corpora = {}
    for doc in docs:
        selected_corpora.setdefault(doc["corpus"], {})[doc["id"]] = (
            environment["corpora"][doc["corpus"]][doc["id"]]
        )
    return freeze_ranker_environment(
        {**environment, "corpora": selected_corpora},
        occurrences_by_document=frozen_occurrences_from_arms(arms),
        floors=floors,
        source_documents={doc["id"]: doc["text"] for doc in docs},
        authoritative_references={doc["id"]: doc["gold_ref"] for doc in docs},
    )


def _utility_close(actual, expected):
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=0.0, abs_tol=_UTILITY_FLOAT_TOLERANCE)
    )


def _frozen_utility_manifest(artifact):
    manifest = artifact.get("threshold_manifest")
    budgets = artifact.get("family_budgets")
    if not isinstance(manifest, dict) or not isinstance(budgets, dict):
        raise SystemExit("utility artifact is missing frozen threshold/family budget state")
    try:
        normalized_manifest = normalize_threshold_manifest(manifest)
        normalized_budgets = normalize_family_budgets(budgets)
    except ValueError as error:
        raise SystemExit(f"utility artifact has invalid threshold manifest: {error}") from None
    if _utility_hash(manifest) != _utility_hash(normalized_manifest):
        raise SystemExit("utility artifact threshold_manifest is not canonically normalized")
    manifest_hash = _utility_hash(normalized_manifest)
    if artifact.get("gate_manifest_hash") != manifest_hash:
        raise SystemExit("utility artifact gate_manifest_hash does not match threshold_manifest")
    expected_manifest_pin = {
        "schema": "qa-threshold-manifest-v1",
        "sha256": manifest_hash,
    }
    if artifact.get("threshold_manifest_pin") != expected_manifest_pin:
        raise SystemExit("utility artifact threshold_manifest_pin does not match threshold_manifest")
    manifest_budgets = normalized_manifest["family_budgets"]
    for family, budget in normalized_budgets.items():
        if not _utility_close(manifest_budgets[family], budget):
            raise SystemExit("utility artifact has inconsistent frozen family budgets")
    try:
        reader_threshold = float(normalized_manifest["reader_threshold"])
        repetitions = int(normalized_manifest["reader_stability_repetitions"])
        option_permutations = int(normalized_manifest["reader_option_permutations"])
        stability_threshold = float(normalized_manifest["reader_stability_threshold"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit("utility artifact is missing frozen reader thresholds") from None
    if (
        not 0.0 <= reader_threshold <= 1.0
        or repetitions < 1
        or option_permutations < 1
        or not 0.0 < stability_threshold <= 1.0
    ):
        raise SystemExit("utility artifact has invalid frozen reader thresholds")
    try:
        cost_budgets = normalize_cost_budgets(normalized_manifest.get("cost_budgets"))
        artifact_cost_budgets = normalize_cost_budgets(artifact.get("cost_budgets"))
    except ValueError as error:
        raise SystemExit(f"utility artifact is missing or has invalid frozen cost budgets: {error}") \
            from None
    if artifact_cost_budgets != cost_budgets:
        raise SystemExit("utility artifact has inconsistent frozen cost budgets")
    return normalized_budgets, {
        "reader_threshold": reader_threshold,
        "repetitions": repetitions,
        "option_permutations": option_permutations,
        "stability_threshold": stability_threshold,
    }, cost_budgets


def _verify_context_anchor(assertion_id, assertion, live_document, occurrence_ids):
    support = assertion.get("expected_action_support")
    if not isinstance(support, dict):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} lacks expected_action_support"
        )
    action_vector = support.get("joint_anchor_action_vector")
    if not isinstance(action_vector, dict) or not action_vector:
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} lacks joint action vector"
        )
    if support.get("joint_anchor_hash") != _utility_hash(action_vector):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has invalid joint anchor hash"
        )
    property_levels = support.get("property_level")
    if not isinstance(property_levels, dict) or support.get("property_levels") is not None:
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has invalid property_level"
        )
    decisions = {
        str(row["decision_id"]): row for row in live_document.get("decisions", [])
    }
    controlled = {
        decision_id: row for decision_id, row in decisions.items()
        if row.get("controlled", True)
    }
    if set(action_vector) != set(controlled):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} joint vector does not cover "
            "controlled decisions"
        )
    occurrences = {
        str(row["occurrence_id"]): row for row in live_document.get("occurrences", [])
    }
    linked_decisions = {str(occurrences[occurrence_id]["decision_id"])
                        for occurrence_id in occurrence_ids}
    if not linked_decisions <= set(controlled):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} links uncontrolled decision"
        )
    if set(property_levels) != linked_decisions:
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has invalid property_level"
        )
    for decision_id, decision in controlled.items():
        actions = {
            str(action["action_id"]): action for action in decision.get("actions", [])
            if action.get("legal", True)
        }
        action = actions.get(str(action_vector[decision_id]))
        if action is None:
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} selects an unknown frozen action"
            )
        mode = action.get("mode")
        if decision_id not in linked_decisions:
            if mode != "keep":
                raise SystemExit(
                    f"utility artifact context assertion {assertion_id} must keep unrelated decision"
                )
            continue
        if mode in {"keep", "placeholder"}:
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} requires a non-placeholder "
                "generalization with matching property support"
            )
        expected_action = coarsest_entailing_legal_action(
            decision, property_levels[decision_id]
        )
        if expected_action is None:
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} requires a non-placeholder "
                "generalization with matching property support"
            )
        if str(action["action_id"]) != str(expected_action["action_id"]):
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} must select the coarsest "
                "entailing action"
            )


def _verify_context_validation(assertion_id, assertion, thresholds):
    validation = (assertion.get("evidence") or {}).get("validation")
    if not isinstance(validation, dict):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} lacks accepted validation evidence"
        )
    scores = validation.get("scores")
    stability = validation.get("stability")
    if not isinstance(scores, dict) or not isinstance(stability, dict):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} lacks accepted validation evidence"
        )
    if (
        stability.get("repetitions") != thresholds["repetitions"]
        or stability.get("option_permutations") != thresholds["option_permutations"]
        or not _utility_close(stability.get("threshold", -1.0), thresholds["stability_threshold"])
    ):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has mismatched frozen reader evidence"
        )
    trials = stability.get("trials")
    expected_trials = [
        (repetition, permutation_index)
        for repetition in range(thresholds["repetitions"])
        for permutation_index in range(thresholds["option_permutations"])
    ]
    if not isinstance(trials, list) or len(trials) != len(expected_trials):
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} lacks complete reader trials"
        )
    passed_trials = []
    for trial, (repetition, permutation_index) in zip(trials, expected_trials):
        trial_scores = trial.get("scores") if isinstance(trial, dict) else None
        if (
            not isinstance(trial_scores, dict)
            or set(trial_scores) != {"original", "representative", "placeholder"}
            or trial.get("repetition") != repetition
            or trial.get("permutation_index") != permutation_index
        ):
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} has invalid reader trial evidence"
            )
        try:
            normalized_scores = {
                key: float(value) for key, value in trial_scores.items()
            }
        except (TypeError, ValueError):
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} has invalid reader trial evidence"
            ) from None
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in normalized_scores.values()):
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} has invalid reader trial evidence"
            )
        trial_passed = (
            normalized_scores["original"] >= thresholds["reader_threshold"]
            and normalized_scores["representative"] >= thresholds["reader_threshold"]
            and normalized_scores["placeholder"] < thresholds["reader_threshold"]
        )
        if trial.get("passed") is not trial_passed:
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} has recomputed validation mismatch"
            )
        passed_trials.append(trial_passed)
    try:
        summary_matches = (
            set(scores) == {"original", "representative", "placeholder"}
            and all(_utility_close(scores[key], trials[0]["scores"][key]) for key in scores)
            and _utility_close(
                stability.get("passing_fraction", -1.0),
                sum(passed_trials) / len(passed_trials),
            )
        )
    except (TypeError, ValueError):
        summary_matches = False
    recomputed_verdict = (
        "accepted" if sum(passed_trials) / len(passed_trials) >= thresholds["stability_threshold"]
        else "unstable" if any(passed_trials) else "unsupported"
    )
    if not summary_matches or validation.get("verdict") != recomputed_verdict:
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has recomputed validation mismatch"
        )
    if recomputed_verdict != "accepted":
        raise SystemExit(
            f"utility artifact context assertion {assertion_id} has unstable reader evidence"
        )


def _verify_document_weights(doc_id, state, rows, family_budgets):
    expected_denominator = sum(family_budgets.values())
    if not _utility_close(state.get("utility_weight_denominator", 0.0), expected_denominator):
        raise SystemExit(f"utility artifact document {doc_id} has invalid denominator")
    expected_present = [family for family in family_budgets if any(
        row.get("family") == family for row in rows
    )]
    expected_missing = [family for family in family_budgets if family not in expected_present]
    if (
        state.get("present_family_budgets") != expected_present
        or state.get("missing_family_budgets") != expected_missing
    ):
        raise SystemExit(f"utility artifact document {doc_id} has invalid family state")
    grouped = {}
    for row in rows:
        family = row.get("family")
        if family not in family_budgets or not row.get("group_id"):
            raise SystemExit(f"utility artifact document {doc_id} has invalid family/group state")
        grouped.setdefault(family, {}).setdefault(str(row["group_id"]), []).append(row)
    for family, budget in family_budgets.items():
        groups = grouped.get(family, {})
        expected_total = budget if groups else 0.0
        try:
            actual_total = sum(float(row["weight"]) for rows in groups.values() for row in rows)
        except (KeyError, TypeError, ValueError):
            raise SystemExit(
                f"utility artifact document {doc_id} has invalid assertion weight"
            ) from None
        if not _utility_close(actual_total, expected_total):
            raise SystemExit(
                f"utility artifact document {doc_id} weights do not match family budget"
            )
        if not groups:
            continue
        expected_group_weight = budget / len(groups)
        for rows in groups.values():
            if not _utility_close(sum(float(row["weight"]) for row in rows), expected_group_weight):
                raise SystemExit(f"utility artifact document {doc_id} has invalid group weight")
            expected_assertion_weight = expected_group_weight / len(rows)
            if any(not _utility_close(row["weight"], expected_assertion_weight) for row in rows):
                raise SystemExit(f"utility artifact document {doc_id} has invalid assertion weight")
    if "weight_groups" not in state:
        raise SystemExit(f"utility artifact document {doc_id} lacks group weight metadata")
    recorded_groups = state["weight_groups"]
    if not isinstance(recorded_groups, dict) or set(recorded_groups) != set(grouped):
        raise SystemExit(f"utility artifact document {doc_id} has invalid group weight metadata")
    for family, groups in grouped.items():
        recorded_family = recorded_groups.get(family)
        if not isinstance(recorded_family, dict) or set(recorded_family) != set(groups):
            raise SystemExit(f"utility artifact document {doc_id} has invalid group weight metadata")
        expected_group_weight = family_budgets[family] / len(groups)
        for group_id, rows in groups.items():
            recorded = recorded_family[group_id]
            if (
                not isinstance(recorded, dict)
                or set(recorded.get("assertion_ids", [])) != {row["assertion_id"] for row in rows}
                or not _utility_close(recorded.get("weight", -1.0), expected_group_weight)
            ):
                raise SystemExit(f"utility artifact document {doc_id} has invalid group weight")


def _verify_call_budget(doc_id, context_count, cost_budgets):
    actual = {
        "base": {
            "remote_round_trips_per_rollout": 1,
            "context_reader_batches_per_rollout": int(context_count > 0),
        },
        "counterfactual": {
            "remote_round_trips_per_selected_pair": 1,
            "context_reader_batches_per_selected_pair": int(context_count > 0),
        },
    }
    for section, fields in actual.items():
        for field, amount in fields.items():
            if amount > cost_budgets[section][field]:
                raise SystemExit(
                    f"utility artifact document {doc_id} exceeds frozen cost budget "
                    f"{section}.{field}"
                )
    return actual


def enforce_utility_artifact_gate(artifact, environment, *, expected_manifest_hash=None):
    """Recompute frozen QA-builder v2 guarantees before training."""
    if artifact.get("artifact_version") != "utility-assertions-v1":
        raise SystemExit("unsupported utility artifact version")
    for pin in (
        "artifact_hash", "task_pin", "builder_pin", "teacher_pin", "reader_pin",
        "scorer_pin", "gate_manifest_hash", "threshold_manifest_pin",
    ):
        if not artifact.get(pin):
            raise SystemExit(f"utility artifact is missing {pin}")
    if (
        expected_manifest_hash is not None
        and artifact.get("gate_manifest_hash") != expected_manifest_hash
    ):
        raise SystemExit("utility artifact does not match the explicit expected manifest hash")
    if artifact["artifact_hash"] != _utility_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    }):
        raise SystemExit("utility artifact artifact_hash does not match its contents")
    if artifact.get("reader_pin") != context_reader_pin():
        raise SystemExit("utility artifact reader_pin does not match the exact live reader pin")
    if artifact.get("task_pin") != AciTaskAdapter.task_pin:
        raise SystemExit("utility artifact task_pin does not match the live task adapter")
    if artifact.get("builder_pin") != builder_pin():
        raise SystemExit("utility artifact builder_pin does not match the live builder")
    teacher = artifact.get("teacher_pin")
    if not isinstance(teacher, dict) or not isinstance(teacher.get("enabled"), bool):
        raise SystemExit("utility artifact has invalid teacher_pin")
    if teacher != relation_teacher_pin(teacher["enabled"]):
        raise SystemExit("utility artifact teacher_pin is not authoritative")
    if artifact.get("scorer_pin") != utility_scorer_pin():
        raise SystemExit("utility artifact scorer_pin does not match the live scorer")
    family_budgets, thresholds, cost_budgets = _frozen_utility_manifest(artifact)
    artifact_documents = artifact.get("documents")
    if not isinstance(artifact_documents, dict):
        raise SystemExit("utility artifact has invalid document coverage")
    if "documents" in environment:
        environment_documents = environment["documents"]
        if (
            isinstance(environment_documents, dict)
            and set(artifact_documents) == set(environment_documents)
        ):
            for doc_id, state in artifact_documents.items():
                live_document = environment_documents[doc_id]
                for field, label in (
                    ("source_hash", "source hash"),
                    ("authoritative_reference_hash", "authoritative reference hash"),
                ):
                    if state.get(field) != live_document.get(field):
                        raise SystemExit(
                            f"utility artifact document {doc_id} {label} does not match "
                            "the frozen environment"
                        )
    if artifact.get("environment_hash") != environment.get("environment_hash"):
        raise SystemExit("utility artifact environment_hash does not match ranker environment")
    if "documents" in environment and (
        not isinstance(environment_documents, dict)
        or set(artifact_documents) != set(environment_documents)
    ):
        raise SystemExit("utility artifact document coverage does not match ranker environment")
    assertions = artifact.get("assertions", {})
    referenced_assertion_ids = set()
    for doc_id, state in artifact_documents.items():
        measurement_state = state.get("measurement_state")
        if measurement_state not in {"measured", "partial", "unsupported", "build_failed"}:
            raise SystemExit(
                f"utility artifact document {doc_id} has invalid measurement_state "
                f"{measurement_state!r}"
            )
        if measurement_state in {"unsupported", "build_failed"}:
            raise SystemExit(
                f"utility artifact document {doc_id} has unusable measurement_state "
                f"{measurement_state!r}"
            )
        assertion_ids = state.get("assertion_ids", [])
        if len(assertion_ids) != len(set(assertion_ids)):
            raise SystemExit(f"utility artifact document {doc_id} repeats assertion ids")
        if not assertion_ids:
            raise SystemExit(
                f"utility artifact document {doc_id} has no accepted assertions"
            )
        missing = [value for value in assertion_ids if value not in assertions]
        if missing:
            raise SystemExit(
                f"utility artifact document {doc_id} has missing assertions: {missing}"
            )
        live_document = environment.get("documents", {}).get(doc_id, {})
        has_frozen_identities = (
            "decisions" in live_document or "occurrences" in live_document
        )
        live_occurrences = {
            str(row["occurrence_id"]): row for row in live_document.get("occurrences", [])
        }
        live_decisions = {
            str(row["decision_id"]): row for row in live_document.get("decisions", [])
        }
        live_controlled_ids = {
            decision_id for decision_id, row in live_decisions.items()
            if row.get("controlled", True)
        }
        live_controlled_order = [
            str(row["decision_id"])
            for row in live_document.get("decisions", [])
            if row.get("controlled", True)
        ]
        artifact_controlled_ids = [str(value) for value in state.get("controlled_decision_ids", [])]
        if len(artifact_controlled_ids) != len(set(artifact_controlled_ids)):
            raise SystemExit(f"utility artifact document {doc_id} repeats controlled decision IDs")
        if has_frozen_identities and artifact_controlled_ids != live_controlled_order:
            raise SystemExit(
                f"utility artifact document {doc_id} controlled decision IDs do not match "
                "the frozen environment"
            )
        has_frozen_decision_keys = has_frozen_identities and any(
            "runtime_type" in row or "canonical_key" in row
            for row in live_decisions.values()
        )
        if has_frozen_decision_keys:
            live_decision_keys = _decision_key_mapping(
                [row for row in live_decisions.values() if row.get("controlled", True)],
                live_controlled_ids,
                doc_id=doc_id,
                error_type=SystemExit,
            )
            artifact_decision_keys = _decision_key_mapping(
                state.get("decision_keys", []),
                set(artifact_controlled_ids),
                doc_id=doc_id,
                error_type=SystemExit,
            )
            if artifact_decision_keys != live_decision_keys:
                raise SystemExit(
                    f"utility artifact document {doc_id} decision key bindings do not match "
                    "the frozen environment"
                )
        live_occurrence_to_decision = {}
        for occurrence_id, row in live_occurrences.items():
            if not row.get("controlled", row.get("decision_id") is not None):
                continue
            decision_id = row.get("decision_id")
            if decision_id is None or str(decision_id) not in live_controlled_ids:
                raise SystemExit(
                    f"utility artifact environment has invalid controlled occurrence "
                    f"{occurrence_id!r}"
                )
            live_occurrence_to_decision[occurrence_id] = str(decision_id)
        artifact_occurrence_to_decision = {
            str(occurrence_id): str(decision_id)
            for occurrence_id, decision_id in state.get("occurrence_to_decision", {}).items()
        }
        if has_frozen_identities and artifact_occurrence_to_decision != live_occurrence_to_decision:
            raise SystemExit(
                f"utility artifact document {doc_id} occurrence-to-decision mapping does not "
                "match the frozen environment"
            )
        rows = []
        semantic_scopes: dict[str, set[str]] = {}
        for assertion_id in assertion_ids:
            if assertion_id in referenced_assertion_ids:
                raise SystemExit(
                    f"utility artifact assertion {assertion_id} appears in multiple documents"
                )
            referenced_assertion_ids.add(assertion_id)
            assertion = assertions[assertion_id]
            rows.append(assertion)
            if assertion.get("assertion_id") != assertion_id:
                raise SystemExit(
                    f"utility artifact assertion {assertion_id} has an unstable row id"
                )
            if assertion.get("doc_id") != doc_id:
                raise SystemExit(
                    f"utility artifact assertion {assertion_id} belongs to document "
                    f"{assertion.get('doc_id')!r}, not {doc_id!r}"
                )
            occurrence_ids = [str(value) for value in assertion.get("occurrence_ids") or []]
            scope = assertion.get("scope")
            if scope == "global" and occurrence_ids:
                raise SystemExit(
                    f"utility artifact global assertion {assertion_id} has occurrence links"
                )
            if scope == "linked" and not occurrence_ids:
                raise SystemExit(
                    f"utility artifact linked assertion {assertion_id} has no occurrence links"
                )
            if scope not in {"linked", "global"}:
                raise SystemExit(
                    f"utility artifact assertion {assertion_id} has invalid scope {scope!r}"
                )
            semantic_key = utility_assertion_semantic_key(assertion)
            prior_scopes = semantic_scopes.setdefault(semantic_key, set())
            if prior_scopes and scope not in prior_scopes:
                raise SystemExit(
                    f"utility artifact document {doc_id} has a duplicate semantic utility "
                    "fact across linked/global scopes"
                )
            prior_scopes.add(scope)
            if live_occurrences:
                missing_occurrences = sorted(set(occurrence_ids) - set(live_occurrences))
                if missing_occurrences:
                    raise SystemExit(
                        f"utility artifact assertion {assertion_id} has unknown occurrence "
                        f"links: {missing_occurrences}"
                    )
                uncontrolled_occurrences = sorted(
                    occurrence_id for occurrence_id in occurrence_ids
                    if not live_occurrences[occurrence_id].get(
                        "controlled",
                        live_occurrences[occurrence_id].get("decision_id") is not None,
                    )
                )
                if uncontrolled_occurrences:
                    raise SystemExit(
                        f"utility artifact assertion {assertion_id} links uncontrolled occurrence "
                        f"identities: {uncontrolled_occurrences}"
                    )
                dangling_decisions = sorted({
                    str(live_occurrences[occurrence_id].get("decision_id"))
                    for occurrence_id in occurrence_ids
                    if str(live_occurrences[occurrence_id].get("decision_id"))
                    not in live_decisions
                })
                if dangling_decisions:
                    raise SystemExit(
                        f"utility artifact assertion {assertion_id} links dangling decision "
                        f"identities: {dangling_decisions}"
                    )
            if assertion.get("status", "accepted") != "accepted":
                if assertion.get("family") == "context":
                    raise SystemExit(
                        f"utility artifact context assertion {assertion_id} is not accepted"
                    )
                raise SystemExit(f"utility artifact assertion {assertion_id} is not accepted")
            if assertion.get("family") != "context":
                continue
            _verify_context_anchor(assertion_id, assertion, live_document, occurrence_ids)
            _verify_context_validation(assertion_id, assertion, thresholds)
        _verify_document_weights(doc_id, state, rows, family_budgets)
        expected_measurement_state = (
            "partial" if state["missing_family_budgets"] else "measured"
        )
        if measurement_state != expected_measurement_state:
            raise SystemExit(
                f"utility artifact document {doc_id} has inconsistent measurement_state"
            )
        authoritative_links = (
            live_occurrence_to_decision
            if has_frozen_identities else artifact_occurrence_to_decision
        )
        linked_decisions = {
            authoritative_links[occurrence_id]
            for row in rows
            if row.get("scope") == "linked"
            for occurrence_id in [str(value) for value in row.get("occurrence_ids") or []]
            if occurrence_id in authoritative_links
        }
        expected_uncovered = [
            decision_id for decision_id in artifact_controlled_ids
            if decision_id not in linked_decisions
        ]
        recorded_uncovered = state.get("uncovered_decision_ids")
        if (
            not isinstance(recorded_uncovered, list)
            or len(recorded_uncovered) != len(set(recorded_uncovered))
            or recorded_uncovered != expected_uncovered
        ):
            raise SystemExit(
                f"utility artifact document {doc_id} has inconsistent uncovered decisions"
            )
        context_count = sum(row.get("family") == "context" for row in rows)
        _verify_call_budget(doc_id, context_count, cost_budgets)
    unassigned = sorted(set(assertions) - referenced_assertion_ids)
    if unassigned:
        raise SystemExit(f"utility artifact has unassigned assertions: {unassigned}")


def qa_utility_preflight_report(artifact, environment):
    """Validate QA-local readiness and describe its fixed call surface without running it."""
    enforce_utility_artifact_gate(artifact, environment)
    _family_budgets, _thresholds, cost_budgets = _frozen_utility_manifest(artifact)
    assertions = artifact["assertions"]
    documents = {}
    total_context = total_delivered = total_uncovered = 0
    for doc_id, state in artifact["documents"].items():
        rows = [assertions[assertion_id] for assertion_id in state["assertion_ids"]]
        context_count = sum(row.get("family") == "context" for row in rows)
        delivered_count = sum(row.get("family") == "delivered" for row in rows)
        total_context += context_count
        total_delivered += delivered_count
        occurrence_to_decision = {
            str(occurrence_id): str(decision_id)
            for occurrence_id, decision_id in state["occurrence_to_decision"].items()
        }
        linked_decisions = {
            occurrence_to_decision[occurrence_id]
            for row in rows if row.get("scope") == "linked"
            for occurrence_id in [str(value) for value in row.get("occurrence_ids") or []]
        }
        uncovered = [
            str(decision_id) for decision_id in state["controlled_decision_ids"]
            if str(decision_id) not in linked_decisions
        ]
        total_uncovered += len(uncovered)
        documents[doc_id] = {
            "measurement_state": state.get("measurement_state", "measured"),
            "accepted_assertion_count": len(rows),
            "context_assertion_count": context_count,
            "delivered_assertion_count": delivered_count,
            "missing_family_budgets": list(state.get("missing_family_budgets", [])),
            "uncovered_decision_ids": uncovered,
            "uncovered_decision_count": len(uncovered),
            "context_reader_batches_per_rollout": int(context_count > 0),
        }
    return {
        "artifact_hash": artifact["artifact_hash"],
        "environment_hash": artifact["environment_hash"],
        "documents": documents,
        "totals": {
            "documents": len(documents),
            "accepted_assertions": total_context + total_delivered,
            "context_assertions": total_context,
            "delivered_assertions": total_delivered,
            "uncovered_decisions": total_uncovered,
        },
        "call_budget": {
            "base": {
                "remote_round_trips_per_rollout": 1,
                "context_reader_batches_per_rollout": {
                    doc_id: row["context_reader_batches_per_rollout"]
                    for doc_id, row in documents.items()
                },
            },
            "counterfactual": {
                "remote_round_trips_per_selected_pair": 1,
                "context_reader_batches_per_selected_pair": {
                    doc_id: row["context_reader_batches_per_rollout"]
                    for doc_id, row in documents.items()
                },
            },
        },
        "cost_budgets": cost_budgets,
        "executed_remote_calls": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.5,0.7")
    ap.add_argument("--floors", default=None,
                    help="override per-type count floors, e.g. 'MISC=1,LOC=200' "
                         "(default: env k_floors, all runtime types 1.0/inert)")
    ap.add_argument("--randomize-floors", action="store_true",
                    help="per-episode log-uniform floor k_T in [k_T/10, 10*k_T], "
                         "log-uniform centered on the default, per type; the "
                         "sampled floor is fed to the policy features (floor-conditioned)")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl-coef", type=float, default=None,
                    help="KL leash coefficient (default: 0.05 surrogate, 0.0 roundtrip; "
                         "an explicit value is always honored)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=16,
                    help="docs loaded per corpus; docs beyond the frozen arms artifact are skipped")
    ap.add_argument("--env", default="data/ranker_env.json",
                    help="ranker environment artifact (default: frozen env; pilot env to retarget)")
    ap.add_argument("--arms", default="data/task_arms_tau0.02.json",
                    help="arms artifact (default: frozen historical; must match --env)")
    ap.add_argument("--policy", choices=["mlp", "encoder"], default="mlp",
                    help="mlp = feature-only RankerPolicy (default); encoder = doc-conditioned "
                         "EncoderPolicy (frozen HF encoder + trainable head)")
    ap.add_argument("--encoder-model", default="answerdotai/ModernBERT-base",
                    help="HF encoder for --policy encoder (frozen; embeds span context once "
                         "per doc at load)")
    ap.add_argument("--smoke", action="store_true", help="2 docs, 2 epochs, G=4")
    ap.add_argument("--reward", choices=["surrogate", "roundtrip"], default="surrogate",
                    help="surrogate = local A/u_qa reward; roundtrip = realized fact recall "
                         "on out_final via roundtrip_batch (hits the proxy)")
    ap.add_argument("--probes", default="data/probes_validated.json",
                    help="validated probes artifact (roundtrip mode only)")
    ap.add_argument("--ladder-probes", default=None,
                    help="optional ladder probe artifact for two-channel roundtrip reward")
    ap.add_argument("--decision-probes", default=None,
                    help="optional decision probe artifact for two-channel roundtrip reward")
    ap.add_argument("--utility-artifact", default=None,
                    help="QA-builder v2 utility artifact; replaces legacy probe filtering")
    ap.add_argument(
        "--expected-utility-manifest-hash",
        default=None,
        help=(
            "required with --utility-artifact; exact gate_manifest_hash expected from the "
            "preregistered canonical threshold manifest"
        ),
    )
    ap.add_argument("--utility-reward-cache", default=None,
                    help="required append-only content-addressed JSONL cache for artifact-backed QA rewards")
    ap.add_argument("--adv", choices=["group", "rloo"], default=None,
                    help="advantage baseline (default: group for surrogate, rloo for roundtrip)")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="entropy bonus (default: 0.0 surrogate, 0.01 roundtrip)")
    ap.add_argument("--rt-workers", type=int, default=8,
                    help="round-trip proxy concurrency (roundtrip mode only)")
    ap.add_argument("--exit-rounds", type=int, default=0,
                    help="expert-iteration rounds before the refiner (0 = off; roundtrip only): "
                         "sample G rollouts/doc, SFT on those strictly beating the floor")
    ap.add_argument("--exit-epochs", type=int, default=10,
                    help="clone_choices SFT epochs per ExIt round")
    ap.add_argument("--cf-frac", type=float, default=0.0,
                    help="exact per-span counterfactual credit (roundtrip only): fraction of "
                         "level-mode spans of a greedy rollout to flip to placeholder and "
                         "re-score for exact per-span advantage (0 = off)")
    ap.add_argument("--force-ungated", action="store_true",
                    help="bypass the round-trip support-scan gate with a loud warning "
                         "(roundtrip mode only)")
    args = ap.parse_args()
    if args.utility_artifact is not None and args.reward != "roundtrip":
        raise SystemExit("--utility-artifact requires --reward roundtrip")
    if args.utility_artifact is not None and not args.expected_utility_manifest_hash:
        raise SystemExit(
            "--utility-artifact requires --expected-utility-manifest-hash with the explicit "
            "expected manifest hash"
        )
    if args.utility_artifact is None and args.expected_utility_manifest_hash is not None:
        raise SystemExit("--expected-utility-manifest-hash requires --utility-artifact")
    assert args.G >= 2, "group-relative advantage needs G >= 2 (std of one reward is NaN)"
    assert 0.0 <= args.cf_frac <= 1.0, "--cf-frac must be in [0, 1]"
    if args.exit_rounds > 0:
        assert args.reward == "roundtrip", \
            "expert-iteration (--exit-rounds) requires --reward roundtrip"
    if args.cf_frac > 0:
        assert args.reward == "roundtrip", \
            "counterfactual credit (--cf-frac) requires --reward roundtrip"
    if args.utility_artifact is not None and args.cf_frac != 0.0:
        raise SystemExit(
            "--cf-frac with --utility-artifact is not implemented; artifact mode currently "
            "supports provisional structured credit and the tested-pair substitution hook only"
        )
    roundtrip = args.reward == "roundtrip"
    if roundtrip and roundtrip_batch is None:
        raise SystemExit("roundtrip reward requires cloak.train.roundtrip (import failed)")
    # mode defaults: explicit flags always win, else surrogate/roundtrip presets (None
    # sentinel -> preset; a passed value is honored verbatim, never overridden)
    adv = args.adv or ("rloo" if roundtrip else "group")
    if roundtrip and adv != "rloo":
        raise SystemExit("--adv group is not implemented for the round-trip loop "
                         "(round-trip uses RLOO); pass --adv rloo or omit it")
    if not roundtrip and adv != "group":
        raise SystemExit("--adv rloo is not implemented for the surrogate loop "
                         "(surrogate uses group-relative advantage); pass --adv group or omit it")
    entropy_coef = (args.entropy_coef if args.entropy_coef is not None
                    else (0.01 if roundtrip else 0.0))
    kl_coef = (args.kl_coef if args.kl_coef is not None
               else (0.0 if roundtrip else 0.05))
    if roundtrip and args.randomize_floors:
        raise SystemExit("--randomize-floors is not implemented for the round-trip loop "
                         "(BC only); run fixed floors")
    torch.manual_seed(args.seed)
    random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = json.loads(Path(args.env).read_text())
    art = load_artifact(args.arms)
    floors = dict(env["k_floors"])
    if args.floors:
        floors.update((t, float(k)) for t, k in
                      (kv.split("=") for kv in args.floors.split(",")))

    # floor-walk teacher legitimately diverges from the artifact's stored tau-walk bc_action
    # (that mask is retired); track whether it happens to coincide so we can still run the
    # exact doc_p reproduction check when it does.
    floor_eq_stored = True
    docs = []
    for corpus, per_doc in env["corpora"].items():
        loaded_documents = {
            document["id"]: document
            for document in load_task_docs(corpus, args.n_docs)
        }
        for doc_id, d in per_doc.items():
            # env may hold more docs than --n-docs loaded texts for (e.g. a small smoke on a
            # full env); only build docs whose text is loaded. load_task_docs is deterministic,
            # so this takes the first n_docs per corpus.
            if doc_id not in loaded_documents or not d["trainable"] or not d["spans"]:
                continue
            loaded_document = loaded_documents[doc_id]
            stored_bc = [s["bc_action"] for s in d["spans"]]
            spans, feats = derive_spans(d["spans"], floors, corpus, device)
            floor_eq_stored &= all(s["bc_action"] == b for s, b in zip(spans, stored_bc))
            docs.append({"id": doc_id, "corpus": corpus, "text": loaded_document["text"],
                         "gold_ref": loaded_document["gold_ref"],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "raw_spans": d["spans"], "spans": spans, "feats": feats,
                         "probes_train": d["probes"]["train"]})
    if roundtrip:
        utility_reward_cache = None
        if args.utility_artifact is not None:
            if not args.utility_reward_cache:
                raise SystemExit("--utility-artifact requires --utility-reward-cache")
            utility_artifact = json.loads(Path(args.utility_artifact).read_text())
            enforce_utility_artifact_gate(
                utility_artifact,
                frozen_training_environment(env, art, docs, floors=floors),
                expected_manifest_hash=args.expected_utility_manifest_hash,
            )
            attached = attach_utility_artifact(docs, utility_artifact)
            utility_reward_cache = UtilityRewardCache(args.utility_reward_cache)
            print(
                f"utility artifact ({args.utility_artifact}): kept {len(attached)}/{len(docs)} "
                "docs with measured utility",
                flush=True,
            )
            print(f"utility reward cache: {args.utility_reward_cache}", flush=True)
            docs = attached
        else:
            # reward uses the validated train-split probes; docs with < 3 distinct facts are
            # excluded from the legacy RL reward.
            from cloak.train.reward import canon
            from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL
            probes_art = json.loads(Path(args.probes).read_text())
            meta = probes_art.get("meta", {})
            if meta.get("rt_model") != RT_MODEL:
                raise SystemExit(
                    f"probe artifact {args.probes} was built for rt_model="
                    f"{meta.get('rt_model')!r} but the reward model is {RT_MODEL!r}; changing the "
                    "reward model re-gates — rebuild probes (scripts/build_probes.py) and re-run "
                    "the support scan before training")
            if "rt_base_url" not in meta or "th" not in meta:
                raise SystemExit(
                    f"probe artifact {args.probes} is missing provenance meta "
                    f"(rt_base_url/th); rebuild probes (scripts/build_probes.py)")
            if meta.get("rt_base_url") != RT_BASE_URL:
                raise SystemExit(
                    f"probe artifact {args.probes} was built against rt_base_url="
                    f"{meta.get('rt_base_url')!r} but the reward endpoint is {RT_BASE_URL!r}; the "
                    "endpoint is part of the reward pin — rebuild probes and re-run the support scan")
            print(f"probe artifact {args.probes}: teacher={meta.get('teacher')!r} "
                  f"th={meta.get('th')} rt_model={meta.get('rt_model')!r}", flush=True)
            probes_all = probes_art["docs"]
            ladder_all = _artifact_docs(args.ladder_probes)
            decision_all = _artifact_docs(args.decision_probes)
            kept = []
            for doc in docs:
                probes_doc = probes_all.get(doc["id"], {})
                ladder_doc = ladder_all.get(doc["id"])
                decision_doc = decision_all.get(doc["id"])
                doc["probes_train"] = probes_doc.get("train", [])
                doc["probes_heldout"] = probes_doc.get("heldout", [])
                if args.ladder_probes is not None:
                    doc["ladder"] = _artifact_entries(ladder_doc)
                if args.decision_probes is not None:
                    doc["decisions"] = _artifact_entries(decision_doc)
                if args.ladder_probes is not None or args.decision_probes is not None:
                    out_hi = _artifact_out_hi(probes_doc, ladder_doc, decision_doc)
                    if out_hi is not None:
                        doc["out_hi"] = out_hi
                    if doc["corpus"] in SCHEMA_CORPORA:
                        doc["schema"] = True
                if len({canon(p["surface"]) for p in doc["probes_train"]}) >= 3:
                    kept.append(doc)
            print(f"roundtrip probes ({args.probes}): kept {len(kept)}/{len(docs)} docs, "
                  f"dropped {len(docs) - len(kept)} with < 3 distinct train facts", flush=True)
            if args.ladder_probes is not None:
                n_ladder = sum(len(d.get("ladder", [])) for d in kept)
                print(f"ladder probes ({args.ladder_probes}): attached {n_ladder} kept rungs",
                      flush=True)
            if args.decision_probes is not None:
                n_decisions = sum(len(d.get("decisions", [])) for d in kept)
                n_out_hi = sum("out_hi" in d for d in kept)
                print(f"decision probes ({args.decision_probes}): attached {n_decisions} "
                      f"decisions; out_hi available for {n_out_hi}/{len(kept)} docs",
                      flush=True)
            docs = kept
    if args.smoke:
        docs, args.epochs, args.G = docs[:2], 2, 4

    encoder_mode = args.policy == "encoder"

    def new_policy():
        return (EncoderPolicy(encoder_name=args.encoder_model).to(device)
                if encoder_mode else RankerPolicy().to(device))

    emb_pol = new_policy() if encoder_mode else None
    if encoder_mode:
        # frozen-encoder span-in-context embeddings, computed ONCE per doc at load and
        # attached as doc["ctx"] (one tensor per span, walk order). Every sample/log_probs
        # call site sets the span's context before scoring (see _ctx_of).
        for doc in docs:
            doc["ctx"] = emb_pol.embed_contexts(
                [span_context(doc["text"], s["start"]) for s in doc["spans"]])
        print(f"encoder policy: {args.encoder_model} embedded contexts for {len(docs)} docs "
              f"(frozen encoder, {sum(len(d['ctx']) for d in docs)} spans)", flush=True)

    all_spans = [s for d in docs for s in d["spans"]]
    n_spans = len(all_spans)
    n_ge2 = sum(len(s["legal"]) >= 2 for s in all_spans)
    n_keep = sum(any(a.get("keep") and i in s["legal"] for i, a in enumerate(s["actions"]))
                 for s in all_spans)
    n_probes = sum(len(d["probes_train"]) for d in docs)
    print(f"train set: docs={len(docs)} spans={n_spans} train-probes={n_probes} "
          f"floors={floors} randomize={args.randomize_floors} device={device}", flush=True)
    print(f"legal-set: spans={n_spans} >=2-legal={n_ge2} keep-original-legal={n_keep}",
          flush=True)
    if floor_eq_stored:
        bad = verify_bc_reproduction(docs, art)
        assert bad == 0, f"{bad} docs fail BC reproduction — assemble != substitute"
        print("BC reproduction verified: assemble(bc) == artifact tau_walk doc_p "
              f"on all {len(docs)} docs", flush=True)
    else:
        # floor-walk teacher differs from the tau-walk reference doc_p; verify the weaker
        # invariants the reproduction check can't cover here. The static per-span floor-walk
        # is NOT injective (unlike the dynamically-masked tau_walk): at high floors several
        # spans collapse onto one generic fill, so assemble() legitimately collides on some
        # docs — the accepted static-teacher / dynamic-mask mismatch (see module docstring).
        # RL rollouts mask collisions dynamically; BC is per-span CE — neither is affected.
        collide = 0
        for doc in docs:
            for s in doc["spans"]:
                assert s["bc_action"] in s["legal"], (doc["id"], s["surface"])
            choice = {s["surface"].lower(): s["actions"][s["bc_action"]] for s in doc["spans"]}
            try:
                assemble(doc["text"], doc["R_walk"], doc["spans"], choice)
            except AssertionError as e:
                if "injectivity" not in str(e):
                    raise
                collide += 1
        print(f"floor-walk teacher diverges from stored tau-walk; verified every bc_action "
              f"legal on all {len(docs)} docs; {collide}/{len(docs)} have a non-injective "
              "static teacher trajectory (accepted mismatch, masked in rollouts)", flush=True)

    if roundtrip:
        if args.utility_artifact is None:
            enforce_support_gate(args.force_ungated, args.probes, args.env)
        from cloak.train.roundtrip import RT_MODEL
        t0 = time.time()
        torch.manual_seed(args.seed)
        policy = emb_pol if encoder_mode else RankerPolicy().to(device)
        policy = behavior_clone(policy, docs, args.bc_epochs, args.lr, device,
                                floors=floors, randomize=args.randomize_floors, seed=args.seed)
        if encoder_mode:
            ref = policy.clone_for_ref()             # shares frozen encoder, deep-copied head
        else:
            ref = RankerPolicy().to(device)
            ref.load_state_dict(policy.state_dict())
        ref.eval()
        log = {"reward": "roundtrip", "rt_model": RT_MODEL, "adv": adv, "floors": floors,
               "randomize_floors": False, "G": args.G, "epochs": args.epochs,
               "n_exit_rounds": args.exit_rounds, "exit_epochs": args.exit_epochs,
               "cf_frac": args.cf_frac,
               "utility_artifact": args.utility_artifact,
               "utility_reward_cache": args.utility_reward_cache,
               "ladder_probes": args.ladder_probes,
               "decision_probes": args.decision_probes,
               "kl_coef": kl_coef, "entropy_coef": entropy_coef, "seed": args.seed,
               "n_docs": len(docs),
               "policy": (f"encoder:{args.encoder_model}" if encoder_mode
                          else "feature-MLP (plan ablation floor)"),
               "exit_rounds": [], "rounds": []}
        # expert-iteration outer loop (after BC, before the RLOO refiner): each round samples
        # G rollouts/doc through the cached round trip and SFTs on the winners strictly beating
        # the floor. --exit-rounds 0 skips it entirely.
        for rnd in range(args.exit_rounds):
            winners, stats = exit_round(docs, policy, G=args.G, rt_workers=args.rt_workers,
                                        seed=args.seed + rnd,
                                        utility_reward_cache=utility_reward_cache)
            clone_choices(policy, [(docs[di]["spans"], docs[di]["feats"], idx,
                                    docs[di].get("ctx")) for di, idx in winners],
                          epochs=args.exit_epochs, lr=args.lr)
            log["exit_rounds"].append({"round": rnd, **stats})
            print(f"[exit] round {rnd}: " +
                  " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
        train_roundtrip(docs, policy, G=args.G, epochs=args.epochs, lr=args.lr,
                        entropy_coef=entropy_coef, kl_coef=kl_coef,
                        ref=(ref if kl_coef > 0 else None), rt_workers=args.rt_workers,
                        seed=args.seed, cf_frac=args.cf_frac, log_rows=log["rounds"],
                        utility_reward_cache=utility_reward_cache)
        # greedy read-out at the env floors; artifact mode reuses the complete reward cache.
        res, phs, greedy_cache = greedy_roundtrip_readout(
            docs,
            policy,
            rt_workers=args.rt_workers,
            utility_reward_cache=utility_reward_cache,
        )
        rs = [r["recall"] or 0.0 for r in res]
        # heldout read-out: SAME greedy rollouts (out_final unchanged), scored on each doc's
        # heldout probes from the validated artifact; docs with empty heldout are skipped.
        # Reward stays train-only — heldout is a generalization spot-check, never optimized.
        held = []
        for doc, r in zip(docs, res):
            hp = doc.get("probes_heldout", [])
            if hp:
                f1s = fact_f1s(r["out_final"], hp)
                held.append(sum(f1s) / len(f1s))
        log["greedy_final"] = {
            "r_train": round(sum(rs) / len(rs), 4) if rs else 0.0,
            "r_heldout": round(sum(held) / len(held), 4) if held else None,
            "ph": round(sum(phs) / len(phs), 4) if phs else 0.0}
        if utility_reward_cache is not None:
            log["greedy_final"].update(greedy_cache)
        log["wall_s"] = round(time.time() - t0, 1)
        tag = "rt" + ("_enc" if encoder_mode else "") + ("_smoke" if args.smoke else "")
        torch.save(policy.state_dict(), f"data/ranker_policy_{tag}.pt")
        Path(f"results/ranker_train_{tag}.json").write_text(json.dumps(log, indent=1))
        print(f"[rt] greedy_final={log['greedy_final']} wall={log['wall_s']}s "
              f"-> results/ranker_train_{tag}.json", flush=True)
        return

    for alpha in [float(a) for a in args.alphas.split(",")]:
        t0 = time.time()
        torch.manual_seed(args.seed)
        # ponytail: encoder mode reloads the frozen encoder per alpha (from HF cache); the
        # head must be fresh each alpha and doc["ctx"] embeddings are reused across alphas.
        policy = new_policy()
        policy = behavior_clone(policy, docs, args.bc_epochs, args.lr, device,
                                floors=floors, randomize=args.randomize_floors, seed=args.seed)
        if encoder_mode:
            ref = policy.clone_for_ref()             # shares frozen encoder, deep-copied head
        else:
            ref = RankerPolicy().to(device)
            ref.load_state_dict(policy.state_dict())
        ref.eval()
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
        log = {"alpha": alpha, "floors": floors, "randomize_floors": args.randomize_floors,
               "G": args.G, "epochs": args.epochs,
               "kl_coef": kl_coef, "seed": args.seed, "n_docs": len(docs),
               "policy": (f"encoder:{args.encoder_model}" if encoder_mode
                          else "feature-MLP (plan ablation floor)"), "rounds": []}

        for epoch in range(args.epochs):
            rng = random.Random(args.seed * 1000 + epoch)
            order = list(range(len(docs)))
            rng.shuffle(order)
            ep = {"r": [], "A": [], "U": [], "ph": [], "kl": []}
            for di in order:
                doc = docs[di]
                if args.randomize_floors:
                    # per-episode log-uniform floor per type, features rebuilt from it
                    span_rows, feats = derive_spans(doc["raw_spans"], sample_floors(floors, rng),
                                                    doc["corpus"], device)
                else:
                    span_rows, feats = doc["spans"], doc["feats"]
                rewards, parts_l, logps_l = [], [], []
                for _ in range(args.G):
                    r, parts, logps = rollout_reward(doc, span_rows, feats, policy, alpha)
                    rewards.append(r)
                    parts_l.append(parts)
                    logps_l.append(logps)
                rt = torch.tensor(rewards)
                adv = (rt - rt.mean()) / (rt.std() + 1e-6)
                pg = -sum(a * torch.stack(lp).sum() for a, lp in zip(adv, logps_l)) / args.G
                kl = 0.0
                for i, (s, f) in enumerate(zip(span_rows, feats)):
                    policy.set_context(_ctx_of(doc, i))
                    ref.set_context(_ctx_of(doc, i))
                    kl = kl + kl_to_ref(policy, ref, f, s["legal"])
                kl = kl / len(span_rows)
                loss = pg + kl_coef * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep["r"].append(rt.mean().item())
                ep["A"].append(sum(p["A"] for p in parts_l) / args.G)
                ep["U"].append(sum(p["U"] for p in parts_l) / args.G)
                ep["ph"].append(sum(p["ph_rate"] for p in parts_l) / args.G)
                ep["kl"].append(kl.item())
            n = len(ep["r"])
            row = {k: round(sum(v) / n, 4) for k, v in ep.items()}
            row.update(epoch=epoch)
            log["rounds"].append(row)
            print(f"[a={alpha}] epoch {epoch}: " +
                  " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"), flush=True)

        # greedy operating point after training (deterministic policy read-out)
        # greedy read-out at the env floors — the deployment operating point; randomization
        # is train-time only (doc["spans"]/doc["feats"] are the fixed env-floor spans)
        greedy = {"r": 0.0, "A": 0.0, "U": 0.0, "ph": 0.0}
        with torch.no_grad():
            for doc in docs:
                r, parts, _ = rollout_reward(doc, doc["spans"], doc["feats"], policy,
                                             alpha, greedy=True)
                greedy["r"] += r / len(docs)
                greedy["A"] += parts["A"] / len(docs)
                greedy["U"] += parts["U"] / len(docs)
                greedy["ph"] += parts["ph_rate"] / len(docs)
        log["greedy_final"] = {k: round(v, 4) for k, v in greedy.items()}
        log["wall_s"] = round(time.time() - t0, 1)

        tag = f"a{alpha}" + ("_enc" if encoder_mode else "") + ("_smoke" if args.smoke else "")
        torch.save(policy.state_dict(), f"data/ranker_policy_{tag}.pt")
        Path(f"results/ranker_train_{tag}.json").write_text(json.dumps(log, indent=1))
        print(f"[a={alpha}] greedy_final={log['greedy_final']} wall={log['wall_s']}s "
              f"-> results/ranker_train_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
