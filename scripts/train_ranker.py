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
import random
import re
import time
from pathlib import Path

import torch

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from cloak.train.ranker import (EncoderPolicy, RankerPolicy, action_features,
                                span_context)
from cloak.train.reward import canon, fact_f1s, stage1_reward, u_qa
from cloak.train.qa_builder import frozen_occurrences_from_arms, freeze_ranker_environment
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
        if not state or state.get("measurement_state") in {"unsupported", "build_failed"}:
            continue
        assertion_ids = state.get("assertion_ids") or []
        if not assertion_ids:
            continue
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
    for s in raw_spans:
        s = dict(s)
        # unknown span types inherit the OTHER floor (default-deny) — never a silent waiver
        k = floors.get(s["type"], floors.get("OTHER", 100.0))
        s["legal"] = [i for i, a in enumerate(s["actions"])
                      if a["mode"] == "placeholder" or a.get("aset", 0) >= k]
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

def sample_rollout(doc, span_rows, feats, policy, greedy=False):
    """Sampling half of a rollout under the DYNAMIC injectivity mask (spec §3.3-1).
    Returns (choice, logps, ph_rate, doc_p, R, legals) — no reward computed here. `legals`
    is the per-span DYNAMIC legal set actually sampled from (walk order), so entropy/KL can
    be scored over the masks the policy really used, not the static floor-legal sets."""
    used: set[str] = set()
    choice, logps, legals, n_level = {}, [], [], 0
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
        logps.append(lp)
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
                    rt_workers, seed, cf_frac=0.0, log_rows=None):
    """RLOO + tie-filter epoch loop against roundtrip_batch. Returns per-epoch stat rows.
    cf_frac > 0 adds an exact per-span counterfactual PG term (counterfactual_terms) on a
    fresh greedy rollout after each doc's group update."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rows = []
    for epoch in range(epochs):
        rng = random.Random(seed * 1000 + epoch)
        order = list(range(len(docs)))
        rng.shuffle(order)
        ep = {"r": [], "ph": [], "ent": [], "ties_skipped": 0, "cf_used": 0}
        for di in order:
            doc = docs[di]
            logps_l, ph_l, legals_l = [], [], []
            jobs = []
            for _ in range(G):
                choice, logps, ph, doc_p, R, legals = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy)
                jobs.append(_roundtrip_job(doc, doc_p, R))
                logps_l.append(logps)
                ph_l.append(ph)
                legals_l.append(legals)
            res = roundtrip_batch(jobs, workers=rt_workers)
            rt = torch.tensor([r["recall"] or 0.0 for r in res])
            ep["r"].append(rt.mean().item())
            ep["ph"].append(sum(ph_l) / G)
            if rt.max() == rt.min():                      # DAPO tie filter
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
            if cf_frac > 0:                             # exact per-span counterfactual credit
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
               "ties_skipped": ep["ties_skipped"], "cf_used": ep["cf_used"]}
        rows.append(row)
        if log_rows is not None:
            log_rows.append(row)
        print(f"[rt] epoch {epoch}: " +
              " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"), flush=True)
    return rows


# ---------- expert iteration (ExIt) outer loop ----------

def exit_round(docs, policy, *, G, rt_workers, seed):
    """One expert-iteration round (spec Phase 2 workhorse): per doc sample G rollouts,
    keep the best strictly beating the floor-walk baseline. Baselines and rollouts all go
    through the cached round trip. Returns (winners, stats)."""
    torch.manual_seed(seed)
    jobs, meta = [], []          # baseline job per doc first, then G rollouts per doc
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
        for _ in range(G):
            choice, _, _, doc_p, R, _ = sample_rollout(doc, doc["spans"], doc["feats"], policy)
            idx = {s["surface"].lower(): next(
                       i for i, a in enumerate(s["actions"])
                       if a is choice[s["surface"].lower()])
                   for s in doc["spans"]}
            job = _roundtrip_job(doc, doc_p, R)
            jobs.append(job)
            meta.append(("roll", di, idx))
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
             "n_winners": len(winners)}
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
    manifest_budgets = manifest.get("family_budgets")
    if not isinstance(manifest_budgets, dict) or set(budgets) != set(manifest_budgets):
        raise SystemExit("utility artifact has inconsistent frozen family budgets")
    try:
        normalized_budgets = {str(family): float(budget) for family, budget in budgets.items()}
    except (TypeError, ValueError):
        raise SystemExit("utility artifact has invalid frozen family budgets") from None
    if not normalized_budgets or any(
        not math.isfinite(budget) or budget < 0.0 for budget in normalized_budgets.values()
    ):
        raise SystemExit("utility artifact has invalid frozen family budgets")
    for family, budget in normalized_budgets.items():
        try:
            matches = _utility_close(manifest_budgets[family], budget)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise SystemExit("utility artifact has inconsistent frozen family budgets")
    try:
        reader_threshold = float(manifest["reader_threshold"])
        repetitions = int(manifest["reader_stability_repetitions"])
        option_permutations = int(manifest["reader_option_permutations"])
        stability_threshold = float(manifest["reader_stability_threshold"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit("utility artifact is missing frozen reader thresholds") from None
    if (
        not 0.0 <= reader_threshold <= 1.0
        or repetitions < 1
        or option_permutations < 1
        or not 0.0 < stability_threshold <= 1.0
    ):
        raise SystemExit("utility artifact has invalid frozen reader thresholds")
    return normalized_budgets, {
        "reader_threshold": reader_threshold,
        "repetitions": repetitions,
        "option_permutations": option_permutations,
        "stability_threshold": stability_threshold,
    }


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
        expected_property = canon(str(property_levels[decision_id]))
        entailed = {canon(str(value)) for value in action.get("entails", [])}
        if mode in {"keep", "placeholder"} or expected_property not in entailed:
            raise SystemExit(
                f"utility artifact context assertion {assertion_id} requires a non-placeholder "
                "generalization with matching property support"
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
        return
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


def enforce_utility_artifact_gate(artifact, environment):
    """Recompute frozen QA-builder v2 guarantees before training."""
    if artifact.get("artifact_version") != "utility-assertions-v1":
        raise SystemExit("unsupported utility artifact version")
    for pin in ("artifact_hash", "task_pin", "builder_pin", "reader_pin", "gate_manifest_hash"):
        if not artifact.get(pin):
            raise SystemExit(f"utility artifact is missing {pin}")
    if artifact["artifact_hash"] != _utility_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    }):
        raise SystemExit("utility artifact artifact_hash does not match its contents")
    family_budgets, thresholds = _frozen_utility_manifest(artifact)
    if artifact.get("environment_hash") != environment.get("environment_hash"):
        live_documents = environment.get("documents", {})
        for doc_id, state in artifact.get("documents", {}).items():
            live_hash = live_documents.get(doc_id, {}).get("environment_document_hash")
            if not live_hash or state.get("environment_document_hash") != live_hash:
                raise SystemExit(
                    f"utility artifact environment_hash/document {doc_id} "
                    "does not match ranker environment"
                )
    assertions = artifact.get("assertions", {})
    referenced_assertion_ids = set()
    for doc_id, state in artifact.get("documents", {}).items():
        assertion_ids = state.get("assertion_ids", [])
        if len(assertion_ids) != len(set(assertion_ids)):
            raise SystemExit(f"utility artifact document {doc_id} repeats assertion ids")
        missing = [value for value in assertion_ids if value not in assertions]
        if missing:
            raise SystemExit(
                f"utility artifact document {doc_id} has missing assertions: {missing}"
            )
        live_document = environment.get("documents", {}).get(doc_id, {})
        live_occurrences = {
            str(row["occurrence_id"]): row for row in live_document.get("occurrences", [])
        }
        live_decisions = {
            str(row["decision_id"]): row for row in live_document.get("decisions", [])
        }
        rows = []
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
            if live_occurrences:
                missing_occurrences = sorted(set(occurrence_ids) - set(live_occurrences))
                if missing_occurrences:
                    raise SystemExit(
                        f"utility artifact assertion {assertion_id} has unknown occurrence "
                        f"links: {missing_occurrences}"
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
    unassigned = sorted(set(assertions) - referenced_assertion_ids)
    if unassigned:
        raise SystemExit(f"utility artifact has unassigned assertions: {unassigned}")


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
    assert args.G >= 2, "group-relative advantage needs G >= 2 (std of one reward is NaN)"
    assert 0.0 <= args.cf_frac <= 1.0, "--cf-frac must be in [0, 1]"
    if args.exit_rounds > 0:
        assert args.reward == "roundtrip", \
            "expert-iteration (--exit-rounds) requires --reward roundtrip"
    if args.cf_frac > 0:
        assert args.reward == "roundtrip", \
            "counterfactual credit (--cf-frac) requires --reward roundtrip"
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
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in per_doc.items():
            # env may hold more docs than --n-docs loaded texts for (e.g. a small smoke on a
            # full env); only build docs whose text is loaded. load_task_docs is deterministic,
            # so this takes the first n_docs per corpus.
            if doc_id not in texts or not d["trainable"] or not d["spans"]:
                continue
            stored_bc = [s["bc_action"] for s in d["spans"]]
            spans, feats = derive_spans(d["spans"], floors, corpus, device)
            floor_eq_stored &= all(s["bc_action"] == b for s, b in zip(spans, stored_bc))
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "raw_spans": d["spans"], "spans": spans, "feats": feats,
                         "probes_train": d["probes"]["train"]})
    if roundtrip:
        if args.utility_artifact is not None:
            utility_artifact = json.loads(Path(args.utility_artifact).read_text())
            enforce_utility_artifact_gate(
                utility_artifact,
                freeze_ranker_environment(
                    env,
                    occurrences_by_document=frozen_occurrences_from_arms(art),
                ),
            )
            attached = attach_utility_artifact(docs, utility_artifact)
            print(
                f"utility artifact ({args.utility_artifact}): kept {len(attached)}/{len(docs)} "
                "docs with measured utility",
                flush=True,
            )
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
                                        seed=args.seed + rnd)
            clone_choices(policy, [(docs[di]["spans"], docs[di]["feats"], idx,
                                    docs[di].get("ctx")) for di, idx in winners],
                          epochs=args.exit_epochs, lr=args.lr)
            log["exit_rounds"].append({"round": rnd, **stats})
            print(f"[exit] round {rnd}: " +
                  " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
        train_roundtrip(docs, policy, G=args.G, epochs=args.epochs, lr=args.lr,
                        entropy_coef=entropy_coef, kl_coef=kl_coef,
                        ref=(ref if kl_coef > 0 else None), rt_workers=args.rt_workers,
                        seed=args.seed, cf_frac=args.cf_frac, log_rows=log["rounds"])
        # greedy read-out at the env floors, scored via one round-trip batch (fixed floor only)
        jobs, phs = [], []
        with torch.no_grad():
            for doc in docs:
                _, _, ph, doc_p, R, _ = sample_rollout(doc, doc["spans"], doc["feats"],
                                                       policy, greedy=True)
                jobs.append(_roundtrip_job(doc, doc_p, R))
                phs.append(ph)
        res = roundtrip_batch(jobs, workers=args.rt_workers)
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
