"""Legacy Stage-1 rankers and the additive lambda-conditioned Ranker-v2 policy.

RankerPolicy remains the training plan's feature-only predecessor (its pre-registered ablation
floor, promoted to v0): no text encoder, with features from the Phase-0 environment artifact.
EncoderPolicy is its frozen-context extension. Their interfaces and checkpoint formats are
historical contracts. ConditionalRankerPolicy is the stable-ID v2 path described by
docs/specs/RL/interactive-ranker-v2.md.

Legacy action features: [is_placeholder, p6, level_index/4, n_levels/4,
                         log10_aset/9, log10_active_floor/9, type one-hot (len(TYPES))]
walk_risk (privacy scalar, ungradable under the utility-only reward) and the corpus one-hot
(train/deploy skew — no corpus label at deployment) were removed 2026-07-08 per the spec's
pre-pilot cleanup notes; the encoder CLS is the sole context channel. walk_risk stays in the
legacy environment artifact as an offline diagnostic, just not fed to the predecessor policy.
"""
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
import torch.nn as nn

from cloak.runtime_types import FINE_DEM_TYPES, RUNTIME_TYPES
from cloak.train.count_reward import CountReward
from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)

TYPES = ["DEM", "DATETIME", "LOC", "QUANTITY", "ORG", "MISC", *FINE_DEM_TYPES, "OTHER"]
N_FEAT = 6 + len(TYPES)


def action_features(span: dict, corpus: str | None = None, floor: float = 1.0) -> torch.Tensor:
    """(n_actions, N_FEAT) feature matrix for one decision span. `floor` is the active
    per-type anonymity-set count floor (the operating knob), fed so the policy can be
    conditioned on it under --randomize-floors. `corpus` is accepted but unused (the corpus
    one-hot was removed) — kept in the signature for call-site stability."""
    t_oh = [0.0] * len(TYPES)
    t_oh[TYPES.index(span["type"]) if span["type"] in TYPES else TYPES.index("OTHER")] = 1.0
    n_lvl = sum(a["mode"] == "level" for a in span["actions"])
    rows = []
    for i, a in enumerate(span["actions"]):
        rows.append([1.0 if a["mode"] == "placeholder" else 0.0,
                     a["p6"], min(i, 4) / 4.0, min(n_lvl, 4) / 4.0,
                     math.log10(max(a.get("aset", 1e9), 1.0)) / 9.0,
                     math.log10(max(floor, 1.0)) / 9.0]
                    + t_oh)
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


@dataclass(frozen=True)
class LambdaProfile:
    name: str
    value: float


@dataclass(frozen=True)
class PolicyState:
    document: RankerDocument
    profile: LambdaProfile
    lambda_magnitude: float
    hidden: torch.Tensor


class FrozenTextEncoder(Protocol):
    embedding_dim: int

    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor: ...


class _ModernBERTFrozenEncoder:
    """Thin frozen CLS adapter used only when no encoder is injected."""

    def __init__(self, encoder_pin: str):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_pin)
        self.model = AutoModel.from_pretrained(encoder_pin)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return self.model(**encoded).last_hidden_state[:, 0]


def _default_frozen_encoder(encoder_pin: str) -> FrozenTextEncoder:
    return _ModernBERTFrozenEncoder(encoder_pin)


class ConditionalRankerPolicy(nn.Module):
    """Lambda-conditioned stable-ID policy with recurrent document state."""

    def __init__(
        self,
        count_reward: CountReward,
        supported_profiles: Sequence[LambdaProfile],
        *,
        max_menu_value: float,
        environment_hash: str,
        encoder_pin: str = "answerdotai/ModernBERT-base",
        encoder: FrozenTextEncoder | None = None,
        hidden_dim: int = 128,
        context_window: int = 256,
        profile_embedding_dim: int = 8,
    ):
        super().__init__()
        profiles = tuple(supported_profiles)
        self._validate_configuration(
            profiles,
            max_menu_value=max_menu_value,
            environment_hash=environment_hash,
            encoder_pin=encoder_pin,
            hidden_dim=hidden_dim,
            context_window=context_window,
            profile_embedding_dim=profile_embedding_dim,
        )
        self.count_reward = count_reward
        self.supported_profiles = profiles
        self.profile_index = {
            profile.name: index for index, profile in enumerate(profiles)
        }
        self.max_menu_value = float(max_menu_value)
        self.environment_hash = environment_hash
        self.encoder_pin = encoder_pin
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        self.profile_embedding_dim = profile_embedding_dim
        self.encoder = encoder if encoder is not None else _default_frozen_encoder(encoder_pin)
        self.encoder_dim = int(self.encoder.embedding_dim)
        if self.encoder_dim <= 0:
            raise ValueError("encoder embedding_dim must be positive")

        self.runtime_types = (*RUNTIME_TYPES, "OTHER")
        self.action_feature_names = (
            "mode:level",
            "mode:keep",
            "mode:placeholder",
            "authored_level_position",
            "number_of_levels",
            *(f"type:{runtime_type}" for runtime_type in self.runtime_types),
            "count_score",
        )
        self.action_feature_dim = len(self.action_feature_names)
        self.action_feature_offset = 2 * self.encoder_dim
        self.scoring_dim = (
            self.action_feature_offset + self.action_feature_dim + hidden_dim
        )

        self.keep_embedding = nn.Parameter(torch.empty(self.encoder_dim))
        self.placeholder_embedding = nn.Parameter(torch.empty(self.encoder_dim))
        nn.init.normal_(self.keep_embedding, std=0.02)
        nn.init.normal_(self.placeholder_embedding, std=0.02)

        self.profile_embeddings = nn.Embedding(len(profiles), profile_embedding_dim)
        nn.init.zeros_(self.profile_embeddings.weight)
        self.film = nn.Linear(
            profile_embedding_dim + 1, 2 * self.scoring_dim
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.head = nn.Sequential(
            nn.Linear(self.scoring_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.gru = nn.GRUCell(
            2 * self.encoder_dim + self.action_feature_dim,
            hidden_dim,
        )

        self._context_embedding_cache: dict[tuple[str, str, str], torch.Tensor] = {}
        self._action_embedding_cache: dict[tuple[str, str], torch.Tensor] = {}

    @staticmethod
    def _validate_configuration(
        profiles: tuple[LambdaProfile, ...],
        *,
        max_menu_value: float,
        environment_hash: str,
        encoder_pin: str,
        hidden_dim: int,
        context_window: int,
        profile_embedding_dim: int,
    ) -> None:
        if not profiles:
            raise ValueError("supported_profiles must not be empty")
        names = [profile.name for profile in profiles]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("lambda profile names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("duplicate lambda profile name")
        values = [profile.value for profile in profiles]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise ValueError("lambda profile values must be finite and nonnegative")
        if (
            isinstance(max_menu_value, bool)
            or not isinstance(max_menu_value, (int, float))
            or not math.isfinite(float(max_menu_value))
            or float(max_menu_value) < 0.0
        ):
            raise ValueError("max_menu_value must be finite and nonnegative")
        if not math.isclose(float(max_menu_value), max(float(value) for value in values)):
            raise ValueError("max_menu_value must equal the supported profile maximum")
        if not environment_hash or not encoder_pin:
            raise ValueError("environment_hash and encoder_pin are required")
        if hidden_dim <= 0 or context_window <= 0 or profile_embedding_dim <= 0:
            raise ValueError("policy dimensions and context_window must be positive")

    @property
    def context_embedding_cache(self) -> Mapping[tuple[str, str, str], torch.Tensor]:
        return MappingProxyType(self._context_embedding_cache)

    @property
    def action_embedding_cache(self) -> Mapping[tuple[str, str], torch.Tensor]:
        return MappingProxyType(self._action_embedding_cache)

    def _parameter_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.head.parameters())
        return parameter.device, parameter.dtype

    def _profile_position(self, profile: LambdaProfile) -> int:
        index = self.profile_index.get(profile.name)
        if index is None or self.supported_profiles[index] != profile:
            raise ValueError(f"unsupported lambda profile: {profile.name}")
        return index

    def _profile_magnitude(self, profile: LambdaProfile) -> float:
        self._profile_position(profile)
        if self.max_menu_value == 0.0:
            return 0.0
        return math.log1p(float(profile.value)) / math.log1p(self.max_menu_value)

    def _embed_frozen(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.encoder_dim), dtype=torch.float32)
        device, _ = self._parameter_device_dtype()
        move = getattr(self.encoder, "to", None)
        if callable(move):
            move(device)
        with torch.no_grad():
            embeddings = self.encoder.embed_texts(tuple(texts))
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("frozen encoder must return a tensor")
        if embeddings.shape != (len(texts), self.encoder_dim):
            raise ValueError(
                "frozen encoder returned an unexpected embedding shape: "
                f"{tuple(embeddings.shape)}"
            )
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("frozen encoder returned non-finite embeddings")
        return embeddings.detach().to(device="cpu", dtype=torch.float32)

    def _prepare_document(self, document: RankerDocument) -> None:
        occurrences = {
            str(occurrence["occurrence_id"]): occurrence
            for occurrence in document.occurrences
        }
        context_keys: list[tuple[str, str, str]] = []
        contexts: list[str] = []
        for decision in document.policy_decisions:
            for occurrence_id in decision.occurrence_ids:
                key = (self.environment_hash, self.encoder_pin, occurrence_id)
                if key in self._context_embedding_cache:
                    continue
                occurrence = occurrences.get(occurrence_id)
                if occurrence is None:
                    raise ValueError(
                        f"missing occurrence {occurrence_id} for {decision.decision_id}"
                    )
                context_keys.append(key)
                contexts.append(span_context(
                    document.text,
                    int(occurrence["start"]),
                    window=self.context_window,
                ))
        if contexts:
            embedded = self._embed_frozen(contexts)
            for key, row in zip(context_keys, embedded):
                self._context_embedding_cache[key] = row

        action_keys: list[tuple[str, str]] = []
        fills: list[str] = []
        for decision in document.policy_decisions:
            for action in decision.actions:
                if action.mode != "level":
                    continue
                if not action.fill:
                    raise ValueError(f"level action {action.action_id} has no fill")
                key = (self.encoder_pin, action.action_id)
                if key in self._action_embedding_cache:
                    continue
                action_keys.append(key)
                fills.append(action.fill)
        if fills:
            embedded = self._embed_frozen(fills)
            for key, row in zip(action_keys, embedded):
                self._action_embedding_cache[key] = row

    def decision_context(
        self, document: RankerDocument, decision: RankerDecision
    ) -> torch.Tensor:
        self._prepare_document(document)
        rows = []
        for occurrence_id in decision.occurrence_ids:
            key = (self.environment_hash, self.encoder_pin, occurrence_id)
            try:
                rows.append(self._context_embedding_cache[key])
            except KeyError as exc:
                raise ValueError(
                    f"missing cached context for {decision.decision_id}:{occurrence_id}"
                ) from exc
        if not rows:
            raise ValueError(f"decision {decision.decision_id} has no occurrences")
        return torch.stack(rows).mean(dim=0)

    @staticmethod
    def _actions_by_id(decision: RankerDecision) -> dict[str, RankerAction]:
        actions = {action.action_id: action for action in decision.actions}
        if len(actions) != len(decision.actions):
            raise ValueError(f"duplicate action id for {decision.decision_id}")
        return actions

    def action_features(
        self,
        decision: RankerDecision,
        action_ids: Sequence[str],
    ) -> torch.Tensor:
        actions_by_id = self._actions_by_id(decision)
        action_ids = tuple(action_ids)
        if not action_ids or len(action_ids) != len(set(action_ids)):
            raise ValueError(f"action_ids must be nonempty and unique for {decision.decision_id}")
        try:
            actions = tuple(actions_by_id[action_id] for action_id in action_ids)
        except KeyError as exc:
            raise ValueError(
                f"unknown action for {decision.decision_id}: {exc.args[0]}"
            ) from exc
        count_scores = self.count_reward.action_scores(
            decision.decision_id, action_ids
        ).detach().to(device="cpu", dtype=torch.float32)
        if count_scores.shape != (len(action_ids),) or not bool(
            torch.isfinite(count_scores).all()
        ):
            raise ValueError(f"invalid count scores for {decision.decision_id}")

        number_of_levels = sum(action.mode == "level" for action in decision.actions)
        type_name = (
            decision.runtime_type
            if decision.runtime_type in self.runtime_types
            else "OTHER"
        )
        type_index = self.runtime_types.index(type_name)
        rows = []
        for action, count_score in zip(actions, count_scores):
            mode = [0.0, 0.0, 0.0]
            try:
                mode[("level", "keep", "placeholder").index(action.mode)] = 1.0
            except ValueError as exc:
                raise ValueError(f"unsupported action mode: {action.mode}") from exc
            level_position = 0.0
            if action.mode == "level":
                if action.authored_level_index is None:
                    raise ValueError(
                        f"level action {action.action_id} lacks authored_level_index"
                    )
                level_position = float(action.authored_level_index) / max(
                    number_of_levels - 1, 1
                )
            type_one_hot = [0.0] * len(self.runtime_types)
            type_one_hot[type_index] = 1.0
            rows.append([
                *mode,
                level_position,
                float(number_of_levels),
                *type_one_hot,
                float(count_score),
            ])
        return torch.tensor(rows, dtype=torch.float32)

    def _action_embeddings(
        self,
        decision: RankerDecision,
        action_ids: Sequence[str],
    ) -> torch.Tensor:
        actions_by_id = self._actions_by_id(decision)
        device, dtype = self._parameter_device_dtype()
        rows = []
        for action_id in action_ids:
            action = actions_by_id[action_id]
            if action.mode == "level":
                key = (self.encoder_pin, action.action_id)
                try:
                    row = self._action_embedding_cache[key]
                except KeyError as exc:
                    raise ValueError(f"missing action embedding for {action.action_id}") from exc
                rows.append(row.to(device=device, dtype=dtype))
            elif action.mode == "keep":
                rows.append(self.keep_embedding)
            elif action.mode == "placeholder":
                rows.append(self.placeholder_embedding)
            else:
                raise ValueError(f"unsupported action mode: {action.mode}")
        return torch.stack(rows)

    def _decision_action_inputs(
        self,
        document: RankerDocument,
        decision: RankerDecision,
        action_ids: Sequence[str],
    ) -> torch.Tensor:
        self._prepare_document(document)
        device, dtype = self._parameter_device_dtype()
        context = self.decision_context(document, decision).to(
            device=device, dtype=dtype
        )
        context = context.unsqueeze(0).expand(len(action_ids), -1)
        action_embeddings = self._action_embeddings(decision, action_ids)
        features = self.action_features(decision, action_ids).to(
            device=device, dtype=dtype
        )
        return torch.cat((context, action_embeddings, features), dim=-1)

    def film_parameters(
        self, profile: LambdaProfile
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = self._profile_position(profile)
        device, dtype = self._parameter_device_dtype()
        profile_index = torch.tensor(index, device=device, dtype=torch.long)
        profile_embedding = self.profile_embeddings(profile_index)
        magnitude = torch.tensor(
            [self._profile_magnitude(profile)], device=device, dtype=dtype
        )
        condition = torch.cat((profile_embedding.to(dtype=dtype), magnitude))
        delta_scale, bias = self.film(condition).chunk(2)
        return 1.0 + delta_scale, bias

    def begin_document(
        self, document: RankerDocument, profile: LambdaProfile
    ) -> PolicyState:
        magnitude = self._profile_magnitude(profile)
        self._prepare_document(document)
        device, dtype = self._parameter_device_dtype()
        return PolicyState(
            document=document,
            profile=profile,
            lambda_magnitude=magnitude,
            hidden=torch.zeros(self.hidden_dim, device=device, dtype=dtype),
        )

    def _validate_state_profile(
        self, state: PolicyState, profile: LambdaProfile
    ) -> None:
        self._profile_position(profile)
        if state.profile != profile:
            raise ValueError("lambda profile changed within document")

    def _validate_state_decision(
        self, state: PolicyState, decision: RankerDecision
    ) -> None:
        matches = [
            row for row in state.document.policy_decisions
            if row.decision_id == decision.decision_id
        ]
        if len(matches) != 1 or matches[0] != decision:
            raise ValueError(
                f"decision {decision.decision_id} does not belong to policy state"
            )

    def log_probs(
        self,
        state: PolicyState,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: LambdaProfile,
    ) -> torch.Tensor:
        self._validate_state_profile(state, profile)
        self._validate_state_decision(state, decision)
        legal_action_ids = tuple(legal_action_ids)
        action_inputs = self._decision_action_inputs(
            state.document, decision, legal_action_ids
        )
        hidden = state.hidden.unsqueeze(0).expand(len(legal_action_ids), -1)
        scoring_inputs = torch.cat((action_inputs, hidden), dim=-1)
        gamma, bias = self.film_parameters(profile)
        logits = self.head(scoring_inputs * gamma + bias).squeeze(-1)
        return torch.log_softmax(logits, dim=0)

    def advance(
        self,
        state: PolicyState,
        decision: RankerDecision,
        action_id: str,
    ) -> PolicyState:
        self._validate_state_decision(state, decision)
        selected = self._decision_action_inputs(
            state.document, decision, (action_id,)
        ).squeeze(0)
        hidden = self.gru(selected, state.hidden)
        return PolicyState(
            document=state.document,
            profile=state.profile,
            lambda_magnitude=state.lambda_magnitude,
            hidden=hidden,
        )

    def sample(
        self,
        state: PolicyState,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: LambdaProfile,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[str, torch.Tensor]:
        legal_action_ids = tuple(legal_action_ids)
        log_probs = self.log_probs(
            state, decision, legal_action_ids, profile
        )
        if greedy:
            selected_index = int(log_probs.argmax())
        else:
            selected_index = int(torch.multinomial(
                log_probs.exp(), 1, generator=generator
            ))
        return legal_action_ids[selected_index], log_probs[selected_index]


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
    assert f[0, 5] == f[1, 5] and f[0, 5] > 0.0            # active-floor feature (idx 5), shared
    pi = RankerPolicy()
    lp = pi.log_probs(f, legal)
    assert lp.shape == (2,) and torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)
    a, alp = pi.sample(f, legal)
    assert a in legal and alp.requires_grad
    a_g, _ = pi.sample(f, legal, greedy=True)
    assert a_g in legal
    print("ranker.py self-check OK")
