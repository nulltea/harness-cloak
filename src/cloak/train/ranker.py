"""Stage-1 ranker policy: MLP over per-action features, masked softmax over the floor-legal set.

This is the training plan's feature-only policy (its pre-registered ablation floor, promoted to
v0): no text encoder, features come entirely from the Phase-0 environment artifact
(data/ranker_env.json). Spec: docs/specs/RL/surrogate-ranker-infiller.md §2 Phase 1. Upgrade
path (plan option 1): frozen-encoder span-in-context embeddings appended to FEATURES.

Action features: [is_placeholder, walk_risk, p6, level_index/4, n_levels/4,
                  log10_aset/9, log10_active_floor/9, type one-hot (7), corpus one-hot (3)]
"""
import math

import torch
import torch.nn as nn

TYPES = ["DEM", "DATETIME", "LOC", "QUANTITY", "ORG", "MISC", "OTHER"]
CORPORA = ["clinical", "enron", "aeslc"]
N_FEAT = 7 + len(TYPES) + len(CORPORA)


def action_features(span: dict, corpus: str, floor: float = 1.0) -> torch.Tensor:
    """(n_actions, N_FEAT) feature matrix for one decision span. `floor` is the active
    per-type anonymity-set count floor (the operating knob), fed so the policy can be
    conditioned on it under --randomize-floors."""
    t_oh = [0.0] * len(TYPES)
    t_oh[TYPES.index(span["type"]) if span["type"] in TYPES else TYPES.index("OTHER")] = 1.0
    c_oh = [0.0] * len(CORPORA)
    c_oh[CORPORA.index(corpus)] = 1.0
    n_lvl = sum(a["mode"] == "level" for a in span["actions"])
    rows = []
    for i, a in enumerate(span["actions"]):
        rows.append([1.0 if a["mode"] == "placeholder" else 0.0,
                     a["walk_risk"], a["p6"], min(i, 4) / 4.0, min(n_lvl, 4) / 4.0,
                     math.log10(max(a.get("aset", 1e9), 1.0)) / 9.0,
                     math.log10(max(floor, 1.0)) / 9.0]
                    + t_oh + c_oh)
    return torch.tensor(rows, dtype=torch.float32)


class RankerPolicy(nn.Module):
    """Scores each action; masked log-softmax over the span's legal set."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def log_probs(self, feats: torch.Tensor, legal: list[int]) -> torch.Tensor:
        """(n_legal,) log-probabilities over the legal actions of one span."""
        scores = self.net(feats).squeeze(-1)          # (n_actions,)
        return torch.log_softmax(scores[legal], dim=0)

    def sample(self, feats: torch.Tensor, legal: list[int],
               greedy: bool = False) -> tuple[int, torch.Tensor]:
        """Returns (action index into span['actions'], log-prob of that action)."""
        lp = self.log_probs(feats, legal)
        j = int(lp.argmax()) if greedy else int(torch.multinomial(lp.exp(), 1))
        return legal[j], lp[j]


if __name__ == "__main__":
    span = {"type": "LOC",
            "actions": [{"fill": "a city in Norway", "mode": "level",
                         "walk_risk": 0.03, "p6": 0.76, "aset": 60.0},
                        {"fill": "a city in Europe", "mode": "level",
                         "walk_risk": 0.003, "p6": 0.52, "aset": 4000.0},
                        {"fill": None, "mode": "placeholder", "walk_risk": 0.0, "p6": 0.0}]}
    legal = [1, 2]
    f = action_features(span, "clinical", floor=100.0)
    assert f.shape == (3, N_FEAT)
    assert f[0, 6] == f[1, 6] and f[0, 6] > 0.0            # active-floor feature, shared
    pi = RankerPolicy()
    lp = pi.log_probs(f, legal)
    assert lp.shape == (2,) and torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)
    a, alp = pi.sample(f, legal)
    assert a in legal and alp.requires_grad
    a_g, _ = pi.sample(f, legal, greedy=True)
    assert a_g in legal
    print("ranker.py self-check OK")
