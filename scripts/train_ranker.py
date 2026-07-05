"""Stage-1 ranker training: contextual bandit, REINFORCE + KL leash, fully local reward.

Implements spec §2 Phase 1 (docs/specs/RL/surrogate-ranker-infiller.md) on the Phase-0
environment (data/ranker_env.json + the arms artifact):

  per doc, per step: sample G level-assignments inside the per-type count-floor mask
  (aset >= k_floors[type]; walk_risk is offline-only) ->
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
from cloak.train.reward import stage1_reward, u_qa

try:  # surrogate-only environments run without the round-trip module
    from cloak.train.roundtrip import roundtrip_batch
except ImportError:
    roundtrip_batch = None

ENV_PATH = Path("data/ranker_env.json")


def _ctx_of(doc, i):
    """Span i's precomputed context embedding, or None (MLP mode has no doc['ctx']).
    set_context(None) is a no-op on RankerPolicy; in encoder mode doc['ctx'] is always set."""
    ctx = doc.get("ctx")
    return None if ctx is None else ctx[i]


# ---------- assembly (rollout -> doc_p, R) ----------

def _case_adjust(fill: str, text: str, start: int) -> str:
    """substitute.py's sentence-start casing, applied at the decision occurrence."""
    prev = text[:start].rstrip()
    sent_start = not prev or prev[-1] in ".!?\n"
    return (fill[0].upper() if sent_start else fill[0].lower()) + fill[1:]


def _cleanup(out: str) -> str:
    """substitute.py's post-substitution cleanup (duplicate articles / 'in in')."""
    out = re.sub(r"\b([Aa]n?|[Tt]he) (?=(?:an?|the)\b)", "", out)
    return re.sub(r"\b[Ii]n (?=in\b)", "", out)


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
        m = re.fullmatch(r"<([A-Z]+)_(\d+)>", e["replacement"])
        if m:
            counters[m.group(1)] = max(counters.get(m.group(1), 0), int(m.group(2)))
    ph_by_surface: dict[str, str] = {}
    used: dict[str, str] = {}
    fills: dict[str, dict] = {}

    def placeholder(skey: str, typ: str) -> str:
        if skey not in ph_by_surface:
            if skey in art_ph:
                ph_by_surface[skey] = art_ph[skey]
            else:
                counters[typ] = counters.get(typ, 0) + 1
                ph_by_surface[skey] = f"<{typ}_{counters[typ]}>"
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
    seen = set()
    for e in sorted(R_walk, key=lambda e: -e["start"]):
        skey = e["surface"].lower()
        # apply the decision only to occurrences the walk treated as quasi (they carry a
        # lattice); a same-surface occurrence typed as a DIRECT identifier keeps its chain
        # token — per-occurrence typing wins, exactly as in substitute()
        if skey in fills and e.get("lattice"):
            rep, act = fills[skey]["replacement"], fills[skey]["action"]
        else:
            rep, act = e["replacement"], e["action"]
        out = out[:e["start"]] + rep + out[e["end"]:]
        # R must cover every APPLIED replacement: mixed-typing surfaces legally map one
        # surface to two replacements (e.g. 'participant'→'a person' AND '<PERSON_1>'),
        # and dropping either breaks inversion of out_p
        if (skey, rep.lower()) not in seen:
            seen.add((skey, rep.lower()))
            R.append({"surface": e["surface"], "type": e["type"],
                      "action": act, "replacement": rep})
    return _cleanup(out), R


def derive_spans(raw_spans, floors, corpus, device):
    """Legal set + floor-walk BC teacher + features from per-type count floors.
    legal = placeholder ∪ {levels with aset >= floor[type]} (walk_risk is offline-only now).
    bc_action = the legal level minimizing (aset, index) — the most specific legal level, by
    min aset not list order (actions["aset"] is not always sorted); placeholder fallback when
    no level is legal. Every span keeps a placeholder so legal is never empty."""
    spans, feats = [], []
    for s in raw_spans:
        s = dict(s)
        # unknown span types inherit the OTHER floor (default-deny) — never a silent waiver
        k = floors.get(s["type"], floors.get("OTHER", 100.0))
        s["legal"] = [i for i, a in enumerate(s["actions"])
                      if a["mode"] == "placeholder" or a.get("aset", 0) >= k]
        ph_idx = next(i for i, a in enumerate(s["actions"]) if a["mode"] == "placeholder")
        s["bc_action"] = min(((a.get("aset", 0), i) for i, a in enumerate(s["actions"])
                              if a["mode"] == "level" and a.get("aset", 0) >= k),
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
    Returns (choice, logps, ph_rate, doc_p, R) — no reward computed here."""
    used: set[str] = set()
    choice, logps, n_level = {}, [], 0
    for i, (s, f) in enumerate(zip(span_rows, feats)):
        policy.set_context(_ctx_of(doc, i))
        legal_dyn = [i for i in s["legal"]
                     if s["actions"][i]["mode"] == "placeholder"
                     or s["actions"][i]["fill"].lower() not in used]
        a_idx, lp = policy.sample(f, legal_dyn, greedy=greedy)
        a = s["actions"][a_idx]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
            n_level += 1
        choice[s["surface"].lower()] = a
        logps.append(lp)
    doc_p, R = assemble(doc["text"], doc["R_walk"], span_rows, choice)
    return choice, logps, 1.0 - n_level / len(span_rows), doc_p, R


def rollout_reward(doc, span_rows, feats, policy, alpha, greedy=False):
    """One rollout with the DYNAMIC injectivity mask (spec §3.3-1: claimed levels are
    unsampleable, not downgraded post-hoc): spans are decided sequentially in walk order;
    a level whose fill is already claimed by a different surface is masked out before
    sampling, so log-probs, A, and ph_rate all describe the action actually executed."""
    choice, logps, ph_rate, doc_p, R = sample_rollout(doc, span_rows, feats, policy, greedy)
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


def counterfactual_terms(doc, policy, choice, logps, base_r, *, frac, rng, rt_workers):
    """Exact per-span credit (spec Phase 2; COMA made exact by reward determinism):
    for a sampled fraction of non-placeholder spans, re-run the round trip with ONLY that
    span flipped to its placeholder; adv_s = base_r - r_cf weights that span's logp.
    Counterfactual doc_p's are cache-friendly (identical across epochs at fixed choices)."""
    cand = [i for i, s in enumerate(doc["spans"])
            if choice[s["surface"].lower()]["mode"] == "level"]
    take = [i for i in cand if rng.random() < frac]
    if not take:
        return 0.0, 0
    jobs = []
    for i in take:
        s = doc["spans"][i]
        cf = dict(choice)
        ph_idx = next(k for k, a in enumerate(s["actions"]) if a["mode"] == "placeholder")
        cf[s["surface"].lower()] = s["actions"][ph_idx]
        doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], cf)
        jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                     "probes": doc["probes_train"]})
    res = roundtrip_batch(jobs, workers=rt_workers)
    term = 0.0
    for i, r in zip(take, res):
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
            logps_l, ph_l = [], []
            jobs = []
            for _ in range(G):
                choice, logps, ph, doc_p, R = sample_rollout(doc, doc["spans"],
                                                             doc["feats"], policy)
                jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                             "probes": doc["probes_train"]})
                logps_l.append(logps)
                ph_l.append(ph)
            res = roundtrip_batch(jobs, workers=rt_workers)
            rt = torch.tensor([r["recall"] or 0.0 for r in res])
            ep["r"].append(rt.mean().item())
            ep["ph"].append(sum(ph_l) / G)
            if rt.max() == rt.min():                      # DAPO tie filter
                ep["ties_skipped"] += 1
                continue
            adv = rloo_advantage(rt)
            pg = -sum(a * torch.stack(lp).sum() for a, lp in zip(adv, logps_l)) / G
            ent = 0.0
            for i, (s, f) in enumerate(zip(doc["spans"], doc["feats"])):
                policy.set_context(_ctx_of(doc, i))
                ent = ent + policy_entropy(policy, f, s["legal"])
            ent = ent / len(doc["spans"])
            loss = pg - entropy_coef * ent
            if kl_coef > 0 and ref is not None:
                kl = 0.0
                for i, (s, f) in enumerate(zip(doc["spans"], doc["feats"])):
                    policy.set_context(_ctx_of(doc, i))
                    ref.set_context(_ctx_of(doc, i))
                    kl = kl + kl_to_ref(policy, ref, f, s["legal"])
                loss = loss + kl_coef * kl / len(doc["spans"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep["ent"].append(ent.item())
            if cf_frac > 0:                             # exact per-span counterfactual credit
                g_choice, g_logps, _, g_doc_p, g_R = sample_rollout(
                    doc, doc["spans"], doc["feats"], policy, greedy=True)
                base_r = roundtrip_batch(
                    [{"corpus": doc["corpus"], "doc_p": g_doc_p, "R": g_R,
                      "probes": doc["probes_train"]}], workers=rt_workers)[0]["recall"] or 0.0
                term, n_cf = counterfactual_terms(doc, policy, g_choice, g_logps, base_r,
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

def _bc_choice_indices(doc) -> dict[str, int]:
    return {s["surface"].lower(): s["bc_action"] for s in doc["spans"]}


def exit_round(docs, policy, *, G, rt_workers, seed):
    """One expert-iteration round (spec Phase 2 workhorse): per doc sample G rollouts,
    keep the best strictly beating the floor-walk baseline. Baselines and rollouts all go
    through the cached round trip. Returns (winners, stats)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    jobs, meta = [], []          # baseline job per doc first, then G rollouts per doc
    per_doc_idx = []
    for di, doc in enumerate(docs):
        # ExIt reference = THE floor-walk baseline via floor_walk_choice (walk-order collision
        # rule resolves colliding fills to placeholder), per spec Phase 2: a rollout is a
        # winner only if it strictly beats the floor-walk round-trip reward. Injective by
        # construction, so assemble() can no longer collide.
        bc_choice = floor_walk_choice(doc["spans"])
        doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], bc_choice)
        jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                     "probes": doc["probes_train"]})
        meta.append(("bc", di, None))
        idxs = []
        for _ in range(G):
            choice, _, _, doc_p, R = sample_rollout(doc, doc["spans"], doc["feats"], policy)
            idx = {s["surface"].lower(): next(
                       i for i, a in enumerate(s["actions"])
                       if a is choice[s["surface"].lower()])
                   for s in doc["spans"]}
            jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                         "probes": doc["probes_train"]})
            meta.append(("roll", di, idx))
            idxs.append(idx)
        per_doc_idx.append(idxs)
    res = roundtrip_batch([j for j in jobs], workers=rt_workers)
    it = iter(res)
    bc_r, rolls = {}, {di: [] for di in range(len(docs))}
    for kind, di, idx in meta:
        r = next(it)["recall"] or 0.0
        if kind == "bc":
            bc_r[di] = r
        else:
            rolls[di].append((r, idx))
    winners, best_rs = [], []
    for di in range(len(docs)):
        if not rolls[di]:
            continue
        best_r, best_idx = max(rolls[di], key=lambda t: t[0])
        best_rs.append(best_r)
        if best_r > bc_r[di]:
            winners.append((di, best_idx))
    bc_vals = list(bc_r.values())
    stats = {"mean_best_r": round(sum(best_rs) / max(len(best_rs), 1), 4),
             "mean_bc_r": round(sum(bc_vals) / len(bc_vals), 4) if bc_vals else None,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.5,0.7")
    ap.add_argument("--floors", default=None,
                    help="override per-type count floors, e.g. 'MISC=1,LOC=200' "
                         "(floor 1 = waiver, legalizes keep-original for that type; "
                         "default: env k_floors, all types 100)")
    ap.add_argument("--randomize-floors", action="store_true",
                    help="per-episode log-uniform floor k_T in [k_T/10, 10*k_T], "
                         "log-uniform centered on the default, per type; the "
                         "sampled floor is fed to the policy features (floor-conditioned)")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=16,
                    help="docs loaded per corpus; docs beyond the frozen arms artifact are skipped")
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
    # mode defaults: explicit flags win, else surrogate/roundtrip presets
    adv = args.adv or ("rloo" if roundtrip else "group")
    entropy_coef = (args.entropy_coef if args.entropy_coef is not None
                    else (0.01 if roundtrip else 0.0))
    kl_coef = (0.0 if roundtrip and args.kl_coef == ap.get_default("kl_coef")
               else args.kl_coef)
    torch.manual_seed(args.seed)
    random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = json.loads(ENV_PATH.read_text())
    art = load_artifact()
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
            if not d["trainable"] or not d["spans"]:
                continue
            stored_bc = [s["bc_action"] for s in d["spans"]]
            spans, feats = derive_spans(d["spans"], floors, corpus, device)
            floor_eq_stored &= all(s["bc_action"] == b for s, b in zip(spans, stored_bc))
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "raw_spans": d["spans"], "spans": spans, "feats": feats,
                         "probes_train": d["probes"]["train"]})
    if roundtrip:
        # reward uses the validated train-split probes; docs with < 3 are excluded from the
        # RL reward (global constraint), never silently kept.
        probes_all = json.loads(Path(args.probes).read_text())
        kept = []
        for doc in docs:
            doc["probes_train"] = probes_all.get(doc["id"], {}).get("train", [])
            if len(doc["probes_train"]) >= 3:
                kept.append(doc)
        print(f"roundtrip probes ({args.probes}): kept {len(kept)}/{len(docs)} docs, "
              f"dropped {len(docs) - len(kept)} with < 3 validated train probes", flush=True)
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
               "randomize_floors": args.randomize_floors, "G": args.G, "epochs": args.epochs,
               "n_exit_rounds": args.exit_rounds, "exit_epochs": args.exit_epochs,
               "cf_frac": args.cf_frac,
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
                _, _, ph, doc_p, R = sample_rollout(doc, doc["spans"], doc["feats"], policy,
                                                    greedy=True)
                jobs.append({"corpus": doc["corpus"], "doc_p": doc_p, "R": R,
                             "probes": doc["probes_train"]})
                phs.append(ph)
        res = roundtrip_batch(jobs, workers=args.rt_workers)
        rs = [r["recall"] or 0.0 for r in res]
        log["greedy_final"] = {"r": round(sum(rs) / len(rs), 4) if rs else 0.0,
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
               "kl_coef": args.kl_coef, "seed": args.seed, "n_docs": len(docs),
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
                loss = pg + args.kl_coef * kl
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
