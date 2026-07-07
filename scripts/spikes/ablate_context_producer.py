"""Context-injection ablation — the offline judge (Level 1 = no RL; Level 2 = joint round trips).

Spec: research-wiki/experiments/context-injection-surface-ablation.md.

Fits ONE shared MLP head per arm over [context_vector ; feats13]; the only variable is the
context producer:
  A0        — no context (feature-only head; reference, not a deploy candidate)
  cls       — frozen ModernBERT-base [CLS] of the span window            (precomputed)
  attn      — frozen ModernBERT token states + a trained attention query (pool trains w/ head)
  biencoder — frozen contrastive sentence embedder, mean-pool            (precomputed)

Level 1 (default, no proxy): per-span action-regret on doc-held-out spans, PRIMARY on non-flat
spans (flat = all legal actions tie on own-recall), with paired-bootstrap CIs over held-out docs
aggregated across all seeds, per corpus.
Level 2 (--level2, hits the served reward): greedily assemble each arm's full held-out-doc
choice, score realized joint R_rt, report paired doc deltas vs floor-walk + the interaction audit
(summed per-span gains vs realized joint gain).

Everything but the producer is matched: same head arch, objective, LR, epochs, seed set, split.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/ablate_context_producer.py \
       --labels results/context_ablation_labels.json --arms A0 cls attn biencoder [--level2]
Self-check (no models/GPU): ... ablate_context_producer.py --selfcheck
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

OUT = Path("results/context_ablation.json")
BIENCODER = "BAAI/bge-small-en-v1.5"
MODERNBERT = "answerdotai/ModernBERT-base"
TAU = 0.1          # soft-target temperature (fixed across arms)
HID = 128
EPOCHS = 300
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]
MAXLEN = 512       # match src/cloak/train/ranker.py EncoderPolicy (Finding 6)


# ---------- metrics (numpy only) ----------
def _ndcg(scores: np.ndarray, gains: np.ndarray) -> float:
    order = np.argsort(-scores)
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains[order] * disc).sum())
    idcg = float((np.sort(gains)[::-1] * disc).sum())
    return dcg / idcg if idcg > 0 else 1.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 1.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def span_metrics(scores: np.ndarray, recalls: np.ndarray) -> dict:
    oracle = recalls.max()
    pred = int(scores.argmax())
    return {"regret": float(oracle - recalls[pred]), "top1": float(recalls[pred] == oracle),
            "ndcg": _ndcg(scores, recalls), "spearman": _spearman(scores, recalls)}


# ---------- producers ----------
def _mlp(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, HID), nn.ReLU(),
                         nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))


class Identity:
    """Frozen precomputed context vector (A0 / cls / biencoder). No trainable params."""
    def __init__(self, ctx):
        self.ctx, self.params = ctx, []
    def reset_params(self):
        pass
    def n_params(self):
        return 0
    def snapshot(self):
        return None
    def restore(self, state):
        pass
    def __call__(self, i):
        return self.ctx[i]


class AttnPool:
    """Trained attention query over frozen (true-length) token states -> pooled context."""
    def __init__(self, states, dim):
        import torch
        self.states, self.dim = states, dim         # states[i]: [Li, dim] (no padding)
        self.q = torch.zeros(dim, requires_grad=True)
        self.reset_params()
        self.params = [self.q]
    def reset_params(self):
        import torch.nn.init as init
        init.normal_(self.q, std=0.02)              # call AFTER torch.manual_seed (Finding 7)
    def n_params(self):
        return int(self.q.numel())
    def snapshot(self):
        return self.q.detach().clone()              # per-seed trained query (Finding N3)
    def restore(self, state):
        import torch
        with torch.no_grad():
            self.q.copy_(state)
    def __call__(self, i):
        s = self.states[i]                          # [Li, dim]
        w = (s @ self.q).softmax(-1)                # [Li]
        return w @ s                                # [dim]


def _modernbert(texts, device, want_tokens):
    """Frozen ModernBERT embeddings: CLS vectors, or per-span true-length token states."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODERNBERT)
    enc = AutoModel.from_pretrained(MODERNBERT).to(device).eval()
    cls, states = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], return_tensors="pt", padding="longest",
                    truncation=True, max_length=MAXLEN).to(device)
            h = enc(**b).last_hidden_state
            if want_tokens:
                m = b["attention_mask"].bool()
                for row in range(h.shape[0]):
                    states.append(h[row][m[row]].cpu())   # [Li, dim], padding stripped
            else:
                cls.append(h[:, 0].cpu())
    dim = enc.config.hidden_size
    return (states if want_tokens else [v for v in torch.cat(cls)]), dim


def build_producer(kind, spans, device):
    """Returns (producer, ctx_dim). Loads models lazily; frozen inputs precomputed once."""
    import torch
    texts = [s["ctx_text"] for s in spans]
    if kind == "A0":
        return Identity([torch.zeros(0) for _ in spans]), 0
    if kind == "cls":
        cls, dim = _modernbert(texts, device, want_tokens=False)
        return Identity(cls), dim
    if kind == "attn":
        states, dim = _modernbert(texts, device, want_tokens=True)
        return AttnPool(states, dim), dim
    if kind == "biencoder":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(BIENCODER, device=str(device))
        emb = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return Identity([torch.tensor(v, dtype=torch.float32) for v in emb]), int(emb.shape[1])
    raise ValueError(kind)


# ---------- fit one arm at one seed ----------
def _feat_tensors(spans):
    import torch
    return ([torch.tensor([a["feats"] for a in s["actions"]], dtype=torch.float32) for s in spans],
            [torch.tensor([a["own_recall"] for a in s["actions"]]) for s in spans])


def fit_eval(spans, train_idx, held_idx, producer, ctx_dim, seed):
    """Returns (held-out rows aligned to held_idx, fitted head). Seed set BEFORE any param init."""
    import torch
    torch.manual_seed(seed)
    if producer.params:
        producer.reset_params()                       # trainable query re-init under this seed
    head = _mlp(ctx_dim + len(spans[0]["actions"][0]["feats"]))
    opt = torch.optim.Adam(list(head.parameters()) + list(producer.params), lr=LR)
    feats, recall = _feat_tensors(spans)

    def scores_of(i):
        f = feats[i]
        if ctx_dim:
            f = torch.cat([producer(i).unsqueeze(0).expand(f.shape[0], -1), f], dim=-1)
        return head(f).squeeze(-1)

    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = 0.0
        for i in train_idx:
            tgt = (recall[i] / TAU).softmax(-1)
            loss = loss - (tgt * scores_of(i).log_softmax(-1)).sum()
        (loss / max(len(train_idx), 1)).backward()
        opt.step()

    rows = []
    with torch.no_grad():
        for i in held_idx:
            m = span_metrics(scores_of(i).numpy(), recall[i].numpy())
            m.update(doc_id=spans[i]["doc_id"], corpus=spans[i]["corpus"],
                     flat=spans[i].get("flat", False), span_idx=i)
            rows.append(m)
    return rows, head


# ---------- aggregation + bootstrap ----------
def _mean(rows, key):
    v = [r[key] for r in rows]
    return round(float(np.mean(v)), 4) if v else None


def _by_corpus(rows, key="regret"):
    out = {c: _mean([r for r in rows if r["corpus"] == c], key)
           for c in sorted({r["corpus"] for r in rows})}
    out["ALL"] = _mean(rows, key)
    return out


def paired_bootstrap(rows_a, rows_b, B=2000, seed=0):
    """Paired Δregret (a - b), 95% CI, resampling held-out DOCS. rows keyed by doc_id."""
    rng = random.Random(seed)
    da, db = {}, {}
    for r in rows_a:
        da.setdefault(r["doc_id"], []).append(r["regret"])
    for r in rows_b:
        db.setdefault(r["doc_id"], []).append(r["regret"])
    docs = [d for d in da if d in db]
    if not docs:
        return {"delta_regret": None, "ci95": None, "beats": False}
    diffs = []
    for _ in range(B):
        samp = [docs[rng.randrange(len(docs))] for _ in docs]
        ma = np.mean([x for d in samp for x in da[d]])
        mb = np.mean([x for d in samp for x in db[d]])
        diffs.append(ma - mb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_regret": round(float(np.mean(diffs)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "beats": bool(hi < 0)}


def paired_by_corpus(rows_a, rows_b):
    """Paired Δregret per corpus + ALL — the winner is gated per corpus (spec), not just globally."""
    corpora = sorted({r["corpus"] for r in rows_a} | {r["corpus"] for r in rows_b})
    out = {c: paired_bootstrap([r for r in rows_a if r["corpus"] == c],
                               [r for r in rows_b if r["corpus"] == c]) for c in corpora}
    out["ALL"] = paired_bootstrap(rows_a, rows_b)
    return out


def seed_average(all_seed_rows):
    """Mean per-span metrics across seeds; returns rows aligned to the held-out span order."""
    avg = []
    for k in range(len(all_seed_rows[0])):
        base = dict(all_seed_rows[0][k])
        for m in ("regret", "top1", "ndcg", "spearman"):
            base[m] = float(np.mean([sr[k][m] for sr in all_seed_rows]))
        avg.append(base)
    return avg


# ---------- Level 2 (joint realized reward; needs the served reward) ----------
def resolve_choice(bspans, chosen):
    """Walk-order collision resolution (train-time rule). chosen: {surface.lower(): action_idx};
    missing spans use bc_action. Later colliders downgrade to placeholder."""
    used, choice = set(), {}
    for s in bspans:
        skey = s["surface"].lower()
        a = s["actions"][chosen.get(skey, s["bc_action"])]
        if a["mode"] == "level" and (a.get("fill") or "").lower() in used:
            a = s["actions"][next(i for i, x in enumerate(s["actions"])
                                  if x["mode"] == "placeholder")]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
        choice[skey] = a
    return choice


def _greedy_choice(bspans, probe_by_surface, spans, head, producer, ctx_dim, floors):
    """Sequential dynamic-mask greedy, mirroring the trainer (train_ranker.py:196-206): walk
    spans in order with a `used` set; a probe span scores ALL its currently-legal action rows
    (the FULL static-legal set minus claimed fills — feats recomputed on the fly, Finding N2/#2)
    and takes the argmax; a non-probe span uses bc_action, downgraded to placeholder if its fill
    is used. Returns (choice dict, {surface: chosen action_idx})."""
    import torch

    from context_ablation_labels import feats13
    used, choice, chosen_idx = set(), {}, {}
    for s in bspans:
        skey = s["surface"].lower()
        if skey in probe_by_surface:
            gi = probe_by_surface[skey]
            cand = [ai for ai in s["legal"]
                    if s["actions"][ai]["mode"] == "placeholder"
                    or (s["actions"][ai].get("fill") or "").lower() not in used]
            floor = floors.get(s["type"], floors.get("OTHER", 100.0))
            f = torch.tensor([feats13(s, ai, floor) for ai in cand], dtype=torch.float32)
            if ctx_dim:
                f = torch.cat([producer(gi).unsqueeze(0).expand(len(cand), -1), f], dim=-1)
            with torch.no_grad():
                aidx = cand[int(head(f).squeeze(-1).argmax())]
        else:
            a0 = s["actions"][s["bc_action"]]
            aidx = s["bc_action"]
            if a0["mode"] == "level" and (a0.get("fill") or "").lower() in used:
                aidx = next(i for i, x in enumerate(s["actions"]) if x["mode"] == "placeholder")
        act = s["actions"][aidx]
        if act["mode"] == "level":
            used.add(act["fill"].lower())
        choice[skey], chosen_idx[skey] = act, aidx
    return choice, chosen_idx


def level2(spans, held_idx, bundles, arm_seed_heads, arm_seed_states, arm_producers, arm_ctxdim,
           floors, workers=6):
    """Per-seed dynamic-mask greedy full-doc assignment -> realized joint R_rt vs floor-walk,
    seed-averaged per doc with per-corpus CIs (Findings N3/#3) + interaction audit. Each seed uses
    its OWN trained producer state (Finding N3/#1)."""
    from train_ranker import assemble

    from cloak.train.roundtrip import roundtrip_batch

    held_docs = sorted({spans[i]["doc_id"] for i in held_idx if spans[i]["doc_id"] in bundles})
    by_doc = {}
    for i in held_idx:
        if spans[i]["doc_id"] in bundles:
            by_doc.setdefault(spans[i]["doc_id"], []).append(i)
    probe_by_surface = {d: {spans[i]["surface"].lower(): i for i in by_doc[d]} for d in held_docs}

    fw = roundtrip_batch([{"corpus": bundles[d]["corpus"], "probes": bundles[d]["probes"],
                           **_assembled(assemble, bundles[d], resolve_choice(bundles[d]["spans"], {}))}
                          for d in held_docs], workers=workers)
    fw_recall = {d: (r["recall"] or 0.0) for d, r in zip(held_docs, fw)}
    own = {i: {a["action_idx"]: a["own_recall"] for a in spans[i]["actions"]} for i in held_idx}
    base_own = {i: next(a["own_recall"] for a in spans[i]["actions"] if a["is_baseline"])
                for i in held_idx}

    results = {}
    for arm, heads in arm_seed_heads.items():
        producer, ctx_dim, states = arm_producers[arm], arm_ctxdim[arm], arm_seed_states[arm]
        jobs, keys, sum_gain = [], [], {}
        for si, head in enumerate(heads):
            producer.restore(states[si])                  # this seed's trained producer state
            for d in held_docs:
                ch, chosen_idx = _greedy_choice(bundles[d]["spans"], probe_by_surface[d],
                                                spans, head, producer, ctx_dim, floors)
                jobs.append({"corpus": bundles[d]["corpus"], "probes": bundles[d]["probes"],
                             **_assembled(assemble, bundles[d], ch)})
                keys.append((si, d))
                # summed per-span gain over spans whose greedy pick was actually measured
                g = [own[i][ci] - base_own[i] for i in by_doc[d]
                     if (ci := chosen_idx[spans[i]["surface"].lower()]) in own[i]]
                sum_gain[(si, d)] = sum(g) if g else 0.0
        outs = roundtrip_batch(jobs, workers=workers)
        by_docgain = {d: [] for d in held_docs}
        by_docsum = {d: [] for d in held_docs}
        for (si, d), o in zip(keys, outs):
            by_docgain[d].append((o["recall"] or 0.0) - fw_recall[d])
            by_docsum[d].append(sum_gain[(si, d)])
        audit = [{"doc_id": d, "corpus": bundles[d]["corpus"],
                  "realized_joint_gain": round(float(np.mean(by_docgain[d])), 4),
                  "summed_span_gain": round(float(np.mean(by_docsum[d])), 4)} for d in held_docs]
        # per-corpus + ALL joint-gain CIs (spec: per-corpus, never averaged)
        gain_rows = {"ALL": [au["realized_joint_gain"] for au in audit]}
        for au in audit:
            gain_rows.setdefault(au["corpus"], []).append(au["realized_joint_gain"])
        by_corpus = {c: {"joint_gain": round(float(np.mean(v)), 4), "ci95": _boot_mean_ci(v),
                         "beats_floorwalk": bool(_boot_mean_ci(v)[0] > 0)}
                     for c, v in gain_rows.items()}
        results[arm] = {"joint_gain_by_corpus": by_corpus, "seeds": len(heads),
                        "interaction_audit": audit}
    return results


def _boot_mean_ci(vals, B=2000, seed=0):
    """95% CI of the mean, resampling docs with replacement."""
    if not vals:
        return [None, None]
    rng = random.Random(seed)
    means = [float(np.mean([vals[rng.randrange(len(vals))] for _ in vals])) for _ in range(B)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _assembled(assemble, bundle, choice):
    doc_p, R = assemble(bundle["text"], bundle["R_walk"], bundle["raw_spans"], choice)
    return {"doc_p": doc_p, "R": R}


def selfcheck():
    import torch
    rng = random.Random(0)
    spans = []
    for n in range(60):
        best = rng.randrange(3)
        acts = []
        for k in range(3):
            f = [0.0, 0.0, 0.0] + [rng.random() for _ in range(10)]
            f[k] = 1.0
            acts.append({"feats": f, "own_recall": 1.0 if k == best else 0.5,
                         "is_baseline": k == 0, "action_idx": k})
        rec = [a["own_recall"] for a in acts]
        spans.append({"doc_id": f"d{n // 6}", "corpus": "syn", "actions": acts,
                      "ctx_text": "", "flat": max(rec) == min(rec), "_best": best})
    idx = list(range(len(spans)))
    tr, he = idx[:40], idx[40:]
    oracle_ctx = Identity([torch.eye(3)[s["_best"]] for s in spans])
    a0 = Identity([torch.zeros(0) for _ in spans])
    r_oracle, _ = fit_eval(spans, tr, he, oracle_ctx, 3, 0)
    r_a0, _ = fit_eval(spans, tr, he, a0, 0, 0)
    reg_o = np.mean([r["regret"] for r in r_oracle])
    reg_0 = np.mean([r["regret"] for r in r_a0])
    print(f"selfcheck: oracle-ctx regret={reg_o:.3f}  A0 regret={reg_0:.3f}")
    assert reg_o < reg_0 - 0.1, (reg_o, reg_0)
    assert reg_o < 0.1, reg_o
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/context_ablation_labels.json")
    ap.add_argument("--arms", nargs="+", default=["A0", "cls", "attn", "biencoder"])
    ap.add_argument("--held-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level2", action="store_true", help="joint realized-reward confirmation (hits :8060)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.loads(Path(args.labels).read_text())
    spans = data["spans"]
    doc_ids = sorted({s["doc_id"] for s in spans})
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)
    n_held = max(1, int(len(doc_ids) * args.held_frac))
    held_docs = set(doc_ids[:n_held])
    tr = [i for i, s in enumerate(spans) if s["doc_id"] not in held_docs]
    he = [i for i, s in enumerate(spans) if s["doc_id"] in held_docs]

    results = {"meta": {"n_spans": len(spans), "n_train": len(tr), "n_held": len(he),
                        "held_docs": len(held_docs), "seeds": SEEDS, "arms": args.arms,
                        "n_flat_held": sum(1 for i in he if spans[i].get("flat")),
                        "maxlen": MAXLEN}, "arms": {}}
    arm_rows, arm_heads, arm_prod, arm_ctxdim, arm_states = {}, {}, {}, {}, {}
    for arm in args.arms:
        producer, ctx_dim = build_producer(arm, spans, device)
        all_seed_rows, seed_heads, seed_states = [], [], []
        for sd in SEEDS:
            rows, head = fit_eval(spans, tr, he, producer, ctx_dim, sd)
            all_seed_rows.append(rows)
            seed_heads.append(head)                       # all seeds kept for Level 2 (Finding 3)
            seed_states.append(producer.snapshot())       # per-seed producer state (Finding N3)
        avg = seed_average(all_seed_rows)                 # per-span, seed-averaged (Finding 4)
        nonflat = [r for r in avg if not r["flat"]]       # PRIMARY on non-flat (Finding 5)
        arm_rows[arm] = nonflat
        arm_heads[arm], arm_prod[arm], arm_ctxdim[arm], arm_states[arm] = \
            seed_heads, producer, ctx_dim, seed_states
        results["arms"][arm] = {
            "producer_trainable_params": producer.n_params(),     # Finding 7
            "primary_nonflat": {"regret": _mean(nonflat, "regret"), "top1": _mean(nonflat, "top1"),
                                "ndcg": _mean(nonflat, "ndcg"), "n": len(nonflat),
                                "regret_by_corpus": _by_corpus(nonflat)},
            "all_spans": {"regret": _mean(avg, "regret"), "top1": _mean(avg, "top1"), "n": len(avg)},
            "flat_rate": round(sum(r["flat"] for r in avg) / max(len(avg), 1), 3)}
        print(f"{arm:10s} non-flat regret={results['arms'][arm]['primary_nonflat']['regret']} "
              f"top1={results['arms'][arm]['primary_nonflat']['top1']} "
              f"params={producer.n_params()} flat_rate={results['arms'][arm]['flat_rate']}", flush=True)

    # references (floor-walk / random-legal) on non-flat held-out spans
    fw, rnd = [], []
    for i in he:
        if spans[i].get("flat"):
            continue
        rec = np.array([a["own_recall"] for a in spans[i]["actions"]])
        base = next(k for k, a in enumerate(spans[i]["actions"]) if a["is_baseline"])
        fw.append({"regret": float(rec.max() - rec[base]), "doc_id": spans[i]["doc_id"],
                   "corpus": spans[i]["corpus"]})
        rnd.append({"regret": float(rec.max() - rec.mean()), "doc_id": spans[i]["doc_id"],
                    "corpus": spans[i]["corpus"]})
    results["references"] = {"floor_walk_regret_by_corpus": _by_corpus(fw),
                             "random_legal_regret_by_corpus": _by_corpus(rnd), "oracle_regret": 0.0}
    results["paired_vs_floor_walk"] = {a: paired_by_corpus(arm_rows[a], fw) for a in args.arms}
    if "A0" in arm_rows:
        results["paired_vs_A0"] = {a: paired_by_corpus(arm_rows[a], arm_rows["A0"])
                                   for a in args.arms if a != "A0"}

    if args.level2:
        bundles = data.get("assembly_bundles", {})
        if bundles:
            results["level2"] = level2(spans, he, bundles, arm_heads, arm_states, arm_prod,
                                       arm_ctxdim, data["meta"]["floors"])
        else:
            results["level2"] = {"error": "no assembly_bundles in labels file"}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
