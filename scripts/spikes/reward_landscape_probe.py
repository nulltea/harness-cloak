"""Diagnostic for the persistent stage-1 NULL: map the reward landscape around the BC init.

Measures, on the 23 training docs (floor env, default floors):
1. ENDPOINTS — r(BC floor-walk init) vs r(all-placeholder) per alpha: does the privacy term
   offer a real climb the policy never took?
2. SWAP DISTRIBUTION — Δr of every single-span level→placeholder swap from the BC point
   (ΔA exact from the p6 table incl. the mean-of-included effect; ΔU nonzero only on the 7
   measured flip probes from probe_flip_floor_env): how many locally-positive steps exist,
   and how big are they vs the measured rollout noise (~0.024)?
3. LEASH COST — mean -log p(placeholder) per span under the trained a=0.3 policy: the KL
   toll (coef 0.05, mean-per-span) of moving toward the all-placeholder region.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/reward_landscape_probe.py
"""
import json

import torch

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from cloak.train.ranker import RankerPolicy
from cloak.train.reward import stage1_reward, u_qa

from train_ranker import assemble, derive_spans

ALPHAS = [0.0, 0.3, 0.5, 0.7]
# the 7 measured downward flips (probe_flip_floor_env.log): (doc, surface) -> f1 loss
FLIP_LOSS = {("aci/D2N001", "hypertension"): 1.0,
             ("aci/D2N002", "polycystic kidneys"): 1.0,
             ("aci/D2N002", "tylenol"): 0.5,
             ("aci/D2N003", "the last 48 hours"): 0.889,
             ("aci/D2N004", "about 10 years ago"): 1.0,
             ("aci/D2N007", "81 milligrams"): 1.0,
             ("aci/D2N008", "the past couple of months"): 0.909}

env = json.loads(open("data/ranker_env.json").read())
art = load_artifact()
floors = dict(env["k_floors"])
policy = RankerPolicy()
policy.load_state_dict(torch.load("data/ranker_policy_a0.3.pt", weights_only=True))
policy.eval()

docs = []
for corpus, per_doc in env["corpora"].items():
    texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
    for doc_id, d in per_doc.items():
        if not d["trainable"] or not d["spans"]:
            continue
        spans, feats = derive_spans(d["spans"], floors, corpus, "cpu")
        docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                     "R_walk": art[corpus][doc_id]["tau_walk"][1], "spans": spans,
                     "feats": feats, "probes": d["probes"]["train"]})

end_bc = {a: [] for a in ALPHAS}
end_ph = {a: [] for a in ALPHAS}
swaps = []       # per-swap dicts with dA and dU
leash_logp = []  # -log p(placeholder) per span under the trained policy

for doc in docs:
    # BC baseline with the trainer's dynamic collision rule
    bc, used = {}, set()
    for s in doc["spans"]:
        a = s["actions"][s["bc_action"]]
        if a["mode"] == "level" and a["fill"].lower() in used:
            a = next(x for x in s["actions"] if x["mode"] == "placeholder")
        if a["mode"] == "level":
            used.add(a["fill"].lower())
        bc[s["surface"].lower()] = a
    doc_p, R = assemble(doc["text"], doc["R_walk"], doc["spans"], bc)
    p6s = [a["p6"] for a in bc.values() if a["mode"] == "level"]
    A_bc = sum(p6s) / len(p6s) if p6s else 0.0
    U_bc, _ = u_qa(doc_p, R, doc["probes"])
    U_bc = U_bc or 0.0

    ph_choice = {s["surface"].lower(): next(x for x in s["actions"]
                                            if x["mode"] == "placeholder")
                 for s in doc["spans"]}
    doc_ph, R_ph = assemble(doc["text"], doc["R_walk"], doc["spans"], ph_choice)
    U_ph, _ = u_qa(doc_ph, R_ph, doc["probes"])
    U_ph = U_ph or 0.0

    for a in ALPHAS:
        end_bc[a].append(stage1_reward(A_bc, U_bc, a))
        end_ph[a].append(stage1_reward(0.0, U_ph, a))

    # single-swap deltas: level -> placeholder, one span at a time
    n_pr = len(doc["probes"])
    for s in doc["spans"]:
        act = bc[s["surface"].lower()]
        if act["mode"] != "level":
            continue
        rest = [p for k, p in ((k, v["p6"]) for k, v in bc.items()
                               if v["mode"] == "level") if True]
        rest = [v["p6"] for k, v in bc.items()
                if v["mode"] == "level" and k != s["surface"].lower()]
        A_new = sum(rest) / len(rest) if rest else 0.0
        dA = A_new - A_bc                       # negative = privacy term improves
        dU = -FLIP_LOSS.get((doc["id"], s["surface"]), 0.0) / n_pr if n_pr else 0.0
        swaps.append({"doc": doc["id"], "surface": s["surface"], "p6": act["p6"],
                      "dA": dA, "dU": dU})

    # leash proxy: trained policy's placeholder log-prob per span
    with torch.no_grad():
        for s, f in zip(doc["spans"], doc["feats"]):
            lp = policy.log_probs(f, s["legal"])
            ph_pos = s["legal"].index(next(i for i in s["legal"]
                                           if s["actions"][i]["mode"] == "placeholder"))
            leash_logp.append(-lp[ph_pos].item())

n = len(docs)
print(f"docs={n} level-swaps={len(swaps)} spans-with-leash={len(leash_logp)}")
print("\nENDPOINTS (mean r over docs):")
for a in ALPHAS:
    print(f"  alpha={a}: r_bc={sum(end_bc[a])/n:.4f}  r_all_placeholder="
          f"{sum(end_ph[a])/n:.4f}  gap={sum(end_ph[a])/n - sum(end_bc[a])/n:+.4f}")
print("\nSINGLE-SWAP Δr (level->placeholder from BC):")
for a in ALPHAS:
    drs = [a_ * 0 for a_ in []]
    drs = [alpha_dr for alpha_dr in
           (a * (-s["dA"]) + (1 - a) * s["dU"] for s in swaps)]
    pos = [d for d in drs if d > 0]
    print(f"  alpha={a}: positive swaps={len(pos)}/{len(drs)} "
          f"mean_pos={sum(pos)/len(pos) if pos else 0:.4f} "
          f"max={max(drs):.4f} min={min(drs):.4f}")
print("\nLEASH: trained a=0.3 policy, -log p(placeholder) per span: "
      f"mean={sum(leash_logp)/len(leash_logp):.3f} nats "
      f"(kl_coef 0.05 x mean-per-span KL term in the loss)")
print("mean-of-included check: swaps where dA > 0 (placeholdering RAISES A): "
      f"{sum(1 for s in swaps if s['dA'] > 0)}/{len(swaps)}")
