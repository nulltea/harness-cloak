"""Stage-1 ranker training: contextual bandit, REINFORCE + KL leash, fully local reward.

Implements spec §2 Phase 1 (docs/specs/RL/surrogate-ranker-infiller.md) on the Phase-0
environment (data/ranker_env.json + the arms artifact):

  per doc, per step: sample G level-assignments inside the tau-legal mask ->
  assemble doc_p/R (injectivity via a DYNAMIC sampling mask: claimed fills unsampleable) ->
  r = alpha*(1 - A_P6) + (1-alpha)*u_qa(train-split probes) ->
  group-relative advantage -> REINFORCE update of the feature policy + KL(pi || pi_0).

pi_0 = behavior clone of the tau-walk's own decisions (never RL from random). Policy =
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
import random
import re
import time
from pathlib import Path

import torch

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from cloak.train.ranker import RankerPolicy, action_features
from cloak.train.reward import stage1_reward, u_qa

ENV_PATH = Path("data/ranker_env.json")


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
    seen_surface = set()
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
        if skey not in seen_surface:
            seen_surface.add(skey)
            R.append({"surface": e["surface"], "type": e["type"],
                      "action": act, "replacement": rep})
    return _cleanup(out), R


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

def rollout_reward(doc, span_rows, feats, policy, alpha, greedy=False):
    """One rollout with the DYNAMIC injectivity mask (spec §3.3-1: claimed levels are
    unsampleable, not downgraded post-hoc): spans are decided sequentially in walk order;
    a level whose fill is already claimed by a different surface is masked out before
    sampling, so log-probs, A, and ph_rate all describe the action actually executed."""
    used: set[str] = set()
    choice, logps, p6s = {}, [], []
    for s, f in zip(span_rows, feats):
        legal_dyn = [i for i in s["legal"]
                     if s["actions"][i]["mode"] == "placeholder"
                     or s["actions"][i]["fill"].lower() not in used]
        a_idx, lp = policy.sample(f, legal_dyn, greedy=greedy)
        a = s["actions"][a_idx]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
            p6s.append(a["p6"])
        choice[s["surface"].lower()] = a
        logps.append(lp)
    doc_p, R = assemble(doc["text"], doc["R_walk"], span_rows, choice)
    A = sum(p6s) / len(p6s) if p6s else 0.0
    U, _ = u_qa(doc_p, R, doc["probes_train"])
    r = stage1_reward(A, U, alpha)
    ph_rate = 1.0 - len(p6s) / len(span_rows)
    return r, {"A": A, "U": U or 0.0, "ph_rate": ph_rate}, logps


# ---------- training ----------

def behavior_clone(policy, docs, epochs, lr, device):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        for doc in docs:
            loss = 0.0
            for s, f in zip(doc["spans"], doc["feats"]):
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
    ap.add_argument("--tau", type=float, default=None,
                    help="re-derive legal sets at a different tau (risks are stored raw)")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 docs, 2 epochs, G=4")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = json.loads(ENV_PATH.read_text())
    art = load_artifact()
    tau = args.tau if args.tau is not None else env["tau"]

    docs = []
    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
        for doc_id, d in per_doc.items():
            if not d["trainable"] or not d["spans"]:
                continue
            spans = []
            for s in d["spans"]:
                s = dict(s)
                s["legal"] = [i for i, a in enumerate(s["actions"])
                              if a["mode"] == "placeholder" or a["walk_risk"] < tau]
                if tau != env["tau"]:
                    # the artifact's bc_action is the walk at env-tau; re-derive the
                    # tau-walk teacher under the requested tau from the stored risks
                    s["bc_action"] = next(
                        (i for i, a in enumerate(s["actions"])
                         if a["mode"] == "level" and a["walk_risk"] < tau),
                        len(s["actions"]) - 1)
                assert s["bc_action"] in s["legal"], (doc_id, s["surface"])
                spans.append(s)
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "spans": spans,
                         "feats": [action_features(s, corpus).to(device) for s in spans],
                         "probes_train": d["probes"]["train"]})
    if args.smoke:
        docs, args.epochs, args.G = docs[:2], 2, 4
    n_spans = sum(len(d["spans"]) for d in docs)
    n_probes = sum(len(d["probes_train"]) for d in docs)
    print(f"train set: docs={len(docs)} spans={n_spans} train-probes={n_probes} "
          f"tau={tau} device={device}", flush=True)
    if tau == env["tau"]:
        bad = verify_bc_reproduction(docs, art)
        assert bad == 0, f"{bad} docs fail BC reproduction — assemble != substitute"
        print("BC reproduction verified: assemble(bc) == artifact tau_walk doc_p "
              f"on all {len(docs)} docs", flush=True)

    for alpha in [float(a) for a in args.alphas.split(",")]:
        t0 = time.time()
        torch.manual_seed(args.seed)
        policy = RankerPolicy().to(device)
        policy = behavior_clone(policy, docs, args.bc_epochs, args.lr, device)
        ref = RankerPolicy().to(device)
        ref.load_state_dict(policy.state_dict())
        ref.eval()
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
        log = {"alpha": alpha, "tau": tau, "G": args.G, "epochs": args.epochs,
               "kl_coef": args.kl_coef, "seed": args.seed, "n_docs": len(docs),
               "policy": "feature-MLP (plan ablation floor)", "rounds": []}

        for epoch in range(args.epochs):
            rng = random.Random(args.seed * 1000 + epoch)
            order = list(range(len(docs)))
            rng.shuffle(order)
            ep = {"r": [], "A": [], "U": [], "ph": [], "kl": []}
            for di in order:
                doc = docs[di]
                rewards, parts_l, logps_l = [], [], []
                for _ in range(args.G):
                    r, parts, logps = rollout_reward(doc, doc["spans"], doc["feats"],
                                                     policy, alpha)
                    rewards.append(r)
                    parts_l.append(parts)
                    logps_l.append(logps)
                rt = torch.tensor(rewards)
                adv = (rt - rt.mean()) / (rt.std() + 1e-6)
                pg = -sum(a * torch.stack(lp).sum() for a, lp in zip(adv, logps_l)) / args.G
                kl = sum(kl_to_ref(policy, ref, f, s["legal"])
                         for s, f in zip(doc["spans"], doc["feats"])) / len(doc["spans"])
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

        tag = f"a{alpha}" + ("_smoke" if args.smoke else "")
        torch.save(policy.state_dict(), f"data/ranker_policy_{tag}.pt")
        Path(f"results/ranker_train_{tag}.json").write_text(json.dumps(log, indent=1))
        print(f"[a={alpha}] greedy_final={log['greedy_final']} wall={log['wall_s']}s "
              f"-> results/ranker_train_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
