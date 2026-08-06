"""Are the authority intervals feasible at all? (CPU, cache-only)

Phase 1 showed both candidate heads failing the same gate from opposite sides:
the linear head satisfied more tie lower bounds (0.556) while violating 92% of
costly-pair upper bounds; the gelu16 head violated fewer upper bounds (0.480)
while satisfying almost no lower bounds (0.090). That pattern is the signature of
CONFLICTING constraints, not of insufficient capacity.

So test it directly: for each decision, is the required interval
[L_j, U_j) non-empty? If L_j > U_j on a material share of decisions, then NO
per-decision alpha can satisfy both, and the additive composition
`z = u + alpha*g*p` is structurally inadequate regardless of the head that
produces alpha. That is the test Sol High proposed for the growing hinge mass,
applied to the authority intervals.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/authority_interval_feasibility.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")


def main() -> None:
    gate = Path("scripts/spikes/authority_interval_gate.py").read_text()
    namespace: dict = {}
    # split on newline+def: the gate file itself contains the literal "def main()"
    # inside a string, and splitting on the bare marker truncates mid-literal.
    exec(compile(gate.split("\ndef main()")[0], "authority_interval_gate", "exec"), namespace)
    data = namespace["_collect"](namespace["_audit_namespace"]())
    rows = data["rows"]

    both = [r for r in rows if r["lower"] is not None and r["upper"] is not None]
    infeasible = [r for r in both if r["lower"] > r["upper"]]
    print(f"decisions with constraints: {len(rows)}  "
          f"documents: {len({r['doc_id'] for r in rows})}")
    print(f"decisions with BOTH bounds: {len(both)}")
    print(f"  INFEASIBLE (lower > upper): {len(infeasible)}"
          f"  ({100 * len(infeasible) / max(1, len(both)):.0f}%)")
    print(f"  documents affected: {len({r['doc_id'] for r in infeasible})}")
    if infeasible:
        gaps = [r["lower"] - r["upper"] for r in infeasible]
        print(f"  gap (lower - upper): median {st.median(gaps):.2f}  max {max(gaps):.2f}")

    lows = [r["lower"] for r in rows if r["lower"] is not None]
    ups = [r["upper"] for r in rows if r["upper"] is not None]
    print(f"\nlower bounds  n={len(lows)}  median {st.median(lows):.2f}"
          f"  p90 {sorted(lows)[int(0.9 * (len(lows) - 1))]:.2f}  max {max(lows):.2f}")
    print(f"upper bounds  n={len(ups)}  median {st.median(ups):.2f}"
          f"  p10 {sorted(ups)[int(0.1 * (len(ups) - 1))]:.2f}  min {min(ups):.2f}")

    alpha = float(torch.nn.functional.softplus(torch.tensor(data["alpha_raw"])))
    print(f"\ntrained global alpha (softplus(alpha_raw)) = {alpha:.3f}")

    # What is the ceiling for a SINGLE global alpha, and for a PERFECT per-decision one?
    total = len(lows) + len(ups)
    best, best_alpha = -1, None
    for step in range(0, 2001):
        candidate = step * 0.05
        satisfied = sum(1 for value in lows if candidate >= value)
        satisfied += sum(1 for value in ups if candidate < value)
        if satisfied > best:
            best, best_alpha = satisfied, candidate
    per_decision = sum(
        1 for r in rows
        if (r["lower"] is None or r["upper"] is None or r["lower"] <= r["upper"])
        for _ in range(int(r["lower"] is not None) + int(r["upper"] is not None))
    )
    print(f"\nCEILINGS (what any head could achieve, ignoring learnability):")
    print(f"  best single global alpha = {best_alpha:.2f} -> {best}/{total} "
          f"constraints ({100 * best / total:.0f}%)")
    print(f"  perfect per-decision alpha -> {per_decision}/{total} "
          f"({100 * per_decision / total:.0f}%)")
    print("\nIf the per-decision ceiling is far below 100%, the additive controller")
    print("cannot express the required authority and no head architecture fixes it.")


if __name__ == "__main__":
    main()
