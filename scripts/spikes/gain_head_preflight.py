"""Phase 0 — saturation-freedom preflight for candidate gain heads (CPU, offline).

The shipped head (`Linear(4608,32) -> Tanh -> Linear(32,1)`) is a constant
function of its input: saturation fraction 1.000, 55/56 bitwise-identical hidden
vectors, alpha spread exactly 0.00e+00 on four independently trained checkpoints
(see gain_field_audit.py). Before changing `enable_controller_gain` and thereby
the architecture pin, check the candidates here on REAL pooled features.

Four checks per candidate, run against the old head as a control:

  1. identity at init   residual must be exactly 0 so the warm start keeps the
                        calibrated global alpha
  2. reachability       the first layer must receive gradient; report the
                        suppression fraction (units whose activation derivative
                        is < 1e-2 -- for GELU, pre-activation < ~-3)
  3. expressivity       fit DISTINCT random per-decision targets. A head that
                        cannot fit arbitrary per-decision values on real features
                        certainly cannot learn per-decision tie ownership. This
                        is a capacity probe, not a learning claim.
  4. saturation-freedom after fitting, suppression fraction must stay low --
                        otherwise the fix moved the saturation rather than
                        removing it (the literature's phrasing).

Candidates (Codex Sol High round 11 revision, reconciled with the literature):
  old-tanh      shipped head, expected to FAIL -- the control
  linear        LayerNorm(affine=False) -> Linear(d,1,bias=False)   diagnostic baseline
  gelu16        LayerNorm(affine=False) -> Linear(d,16) -> GELU -> Linear(16,1,bias=False)
  gelu32        same at width 32 -- escalation only

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/gain_head_preflight.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

OUT = Path("results/ranker_v2/architecture/count_to_gain")
SUPPRESSION_TOLERANCE = 0.2      # ENGINEERING threshold, not literature-derived:
                                 # nothing in the cited work fixes a number here. It is
                                 # chosen far below the measured failure (0.99) and far
                                 # above a healthy head (0.02-0.03), so the verdict does
                                 # not turn on its exact value.
DERIVATIVE_FLOOR = 1e-2


def _candidates(input_dim: int) -> dict[str, nn.Module]:
    def zero_final(head: nn.Module) -> nn.Module:
        final = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
        return head

    heads = {
        # control: exactly what is shipped today
        "old-tanh": nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(), nn.Linear(32, 1),
        ),
        "linear": nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, 1, bias=False),
        ),
    }
    for width in (16, 32):
        # Sol High's placement: norm on the INPUT only.
        heads[f"gelu{width}-inputnorm"] = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, 1, bias=False),
        )
        # The literature's placement: gain-free norm on the PRE-ACTIVATION, which is
        # the only variant invariant to the first layer's weight growth -- the cause
        # actually measured (|W1| grew 6.2x during training).
        heads[f"gelu{width}-prenorm"] = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, width),
            nn.LayerNorm(width, elementwise_affine=False),
            nn.GELU(),
            nn.Linear(width, 1, bias=False),
        )
    return {name: zero_final(head) for name, head in heads.items()}


def _hidden_pre_activation(head: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """Pre-activation feeding the nonlinearity, or None for a linear head."""
    activation_index = next(
        (i for i, m in enumerate(head) if isinstance(m, nn.Tanh | nn.GELU)), None
    )
    if activation_index is None:
        return None
    value = x
    for module in list(head)[:activation_index]:
        value = module(value)
    return value


def _suppression_fraction(head: nn.Module, inputs: torch.Tensor) -> float | None:
    """Fraction of hidden units whose local activation derivative is below the floor."""
    pre = _hidden_pre_activation(head, inputs)
    if pre is None:
        return None
    pre = pre.detach()
    if any(isinstance(m, nn.Tanh) for m in head):
        derivative = 1.0 - torch.tanh(pre) ** 2
    else:                                   # GELU derivative, tanh-free approximation
        cdf = 0.5 * (1.0 + torch.erf(pre / 2 ** 0.5))
        pdf = torch.exp(-0.5 * pre ** 2) / (2 * torch.pi) ** 0.5
        derivative = cdf + pre * pdf
    return float((derivative.abs() < DERIVATIVE_FLOOR).float().mean())


def _spread(values: torch.Tensor) -> dict:
    flat = values.detach().flatten()
    unique = {round(float(v), 12) for v in flat}
    return {
        "min": float(flat.min()), "max": float(flat.max()),
        "spread": float(flat.max() - flat.min()),
        "std": float(flat.std()) if flat.numel() > 1 else 0.0,
        "distinct_values": len(unique),
        "n": int(flat.numel()),
    }


def _pooled_features() -> torch.Tensor:
    """Real pooled gain-head inputs, path-faithful (state advanced per decision)."""
    source = Path("scripts/spikes/gain_field_audit.py").read_text().split("def main()")[0]
    namespace: dict = {}
    exec(compile(source, "gain_field_audit", "exec"), namespace)
    documents, profiles, count_state = namespace["_environment"]()
    checkpoint = torch.load(
        OUT / "coupled-s47.pt", map_location="cpu", weights_only=False,
    )
    policy = namespace["_policy"](
        documents, profiles, count_state, checkpoint["policy_state_dict"],
    )
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory

    profile, rows = profiles[-1], []
    with torch.no_grad():
        for document in documents:
            # Behaviour-path-faithful: advance along the checkpoint's own greedy
            # trajectory. An earlier version advanced with `actions[0]`, which is
            # history-aware but visits states the policy never chooses.
            greedy = sample_trajectory(
                policy, document, profile, greedy=True, generator=None,
            )
            replayed = replay_trajectory(policy, document, greedy, profile)
            state = policy.begin_document(document, profile)
            for decision, step in zip(
                document.policy_decisions, replayed.steps, strict=True,
            ):
                rows.append(namespace["_pooled_input"](policy, state, decision))
                state = policy.advance(state, decision, step.selected_action_id)
    return torch.stack(rows)


def evaluate(name: str, head: nn.Module, features: torch.Tensor) -> dict:
    # 1. identity at init
    with torch.no_grad():
        initial = head(features).squeeze(-1)
    identity_ok = bool(torch.all(initial == 0.0))

    # 2. reachability, before any fitting
    first = [m for m in head.modules() if isinstance(m, nn.Linear)][0]
    head.zero_grad(set_to_none=True)
    head(features).squeeze(-1).sum().backward()
    # a zero-initialised final layer blocks gradient to earlier layers by design at
    # step 0, so reachability is judged AFTER fitting (below) as well
    reach_at_init = float(first.weight.grad.abs().sum()) if first.weight.grad is not None else 0.0
    suppression_at_init = _suppression_fraction(head, features)

    # 3. expressivity: distinct per-decision targets
    generator = torch.Generator().manual_seed(17)
    targets = torch.randn(features.shape[0], generator=generator)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    losses = []
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(features).squeeze(-1), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    with torch.no_grad():
        fitted = head(features).squeeze(-1)

    # 4. saturation-freedom after fitting, and reachability after fitting
    head.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(head(features).squeeze(-1), targets).backward()
    reach_after = float(first.weight.grad.abs().sum()) if first.weight.grad is not None else 0.0
    suppression_after = _suppression_fraction(head, features)

    correlation = float(torch.corrcoef(torch.stack([fitted, targets]))[0, 1])
    return {
        "head": name,
        "parameters": sum(p.numel() for p in head.parameters()),
        "identity_at_init": identity_ok,
        "residual_at_init": _spread(initial),
        "first_layer_grad_at_init": reach_at_init,
        "first_layer_grad_after_fit": reach_after,
        "suppression_at_init": suppression_at_init,
        "suppression_after_fit": suppression_after,
        "fit_loss_first": losses[0], "fit_loss_last": losses[-1],
        "target_correlation": correlation,
        "residual_after_fit": _spread(fitted),
    }


SEEDS = (17, 29, 47)


def main() -> None:
    features = _pooled_features()
    print(f"pooled features: {tuple(features.shape)}  "
          f"L2 norm mean {float(features.norm(dim=-1).mean()):.1f}")
    print(f"initialisation seeds: {SEEDS}\n")
    reports = []
    names = list(_candidates(features.shape[-1]))
    for name in names:
        # Three seeds: a single uncontrolled init cannot support a width comparison.
        runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            runs.append(evaluate(name, _candidates(features.shape[-1])[name], features))
        report = dict(runs[0])
        report["seeds"] = list(SEEDS)
        for key in ("suppression_after_fit", "fit_loss_last", "target_correlation"):
            values = [r[key] for r in runs if r[key] is not None]
            report[f"{key}_mean"] = (sum(values) / len(values)) if values else None
            report[f"{key}_max"] = max(values) if values else None
            report[f"{key}_min"] = min(values) if values else None
        report["distinct_min"] = min(
            r["residual_after_fit"]["distinct_values"] for r in runs
        )
        report["per_seed"] = runs
        reports.append(report)
        supp_worst = report["suppression_after_fit_max"]
        verdict = []
        if not all(r["identity_at_init"] for r in report["per_seed"]):
            verdict.append("FAIL identity")
        if report["distinct_min"] < features.shape[0]:
            verdict.append(
                f"FAIL expressivity (duplicates: {report['distinct_min']}/{features.shape[0]})"
            )
        if supp_worst is not None and supp_worst > SUPPRESSION_TOLERANCE:
            verdict.append(
                f"FAIL saturation (worst seed {supp_worst:.3f} > {SUPPRESSION_TOLERANCE})"
            )
        if any(r["first_layer_grad_after_fit"] <= 0.0 for r in report["per_seed"]):
            verdict.append("FAIL unreachable first layer")
        print(f"=== {name} ({report['parameters']:,} params) ===")
        supp_mean = report["suppression_after_fit_mean"]
        print(f"  suppression after fit  mean "
              f"{'n/a' if supp_mean is None else f'{supp_mean:.3f}'}"
              f"  worst {'n/a' if supp_worst is None else f'{supp_worst:.3f}'}")
        print(f"  fit MSE (mean over seeds) {report['fit_loss_last_mean']:.4f}"
              f"   corr {report['target_correlation_mean']:+.3f}")
        print(f"  distinct residuals (worst seed) {report['distinct_min']}/{features.shape[0]}")
        print(f"  VERDICT: {'PASS' if not verdict else ' | '.join(verdict)}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gain-head-preflight.json").write_text(json.dumps(reports, indent=1))
    print(f"wrote {OUT / 'gain-head-preflight.json'}")


if __name__ == "__main__":
    main()
