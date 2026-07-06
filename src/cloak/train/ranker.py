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
CORPORA = ["clinical", "enron", "aeslc", "lexsum", "wikibio"]
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

    def set_context(self, ctx_emb=None):
        """No-op: the feature-only policy is not doc-conditioned. Lets trainer call sites
        set span context unconditionally (EncoderPolicy overrides this)."""
        pass


def span_context(text: str, start: int, window: int = 256) -> str:
    """±window chars around the span start, whitespace-normalized — the encoder's view."""
    lo, hi = max(0, start - window), min(len(text), start + window)
    return " ".join(text[lo:hi].split())


class EncoderPolicy(nn.Module):
    """Doc-conditioned ranker policy: score(action) = MLP([ctx_emb ; action_feats]).
    The encoder is FROZEN (feature extractor; embeddings precomputed per span at load) —
    only the head trains, so optimization cost matches the MLP policy. Same sample/log_probs
    contract as RankerPolicy, plus set_context/embed_contexts.
    ponytail: no fine-tuning path; unfreeze via a separate task if capacity still binds."""

    def __init__(self, encoder_name: str = "answerdotai/ModernBERT-base",
                 feat_dim: int = N_FEAT, hid: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        enc_dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(enc_dim + feat_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))
        self._ctx = None

    @torch.no_grad()
    def embed_contexts(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), enc_dim) CLS embeddings; frozen encoder, batched, no grad."""
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
        enc = {k: v.to(next(self.head.parameters()).device) for k, v in enc.items()}
        return self.encoder(**enc).last_hidden_state[:, 0]      # CLS per text

    def set_context(self, ctx_emb: torch.Tensor):
        """Set the current span's precomputed context embedding (shape [enc_dim])."""
        self._ctx = ctx_emb

    def clone_for_ref(self):
        """KL reference: shares the SAME frozen encoder object, deep-copies the trainable
        head (so the reference head is decoupled from the policy head)."""
        import copy
        ref = EncoderPolicy.__new__(EncoderPolicy)
        nn.Module.__init__(ref)
        ref.tok = self.tok
        ref.encoder = self.encoder                 # same frozen object, no reload
        ref.head = copy.deepcopy(self.head)
        ref._ctx = None
        return ref

    def log_probs(self, feats: torch.Tensor, legal: list[int]) -> torch.Tensor:
        """(n_legal,) log-probabilities over the legal actions of one span."""
        assert self._ctx is not None, "call set_context(ctx_emb) before scoring"
        ctx = self._ctx.unsqueeze(0).expand(len(legal), -1)
        x = torch.cat([ctx, feats[legal]], dim=-1)
        return torch.log_softmax(self.head(x).squeeze(-1), dim=0)

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
