"""Round-trip support scan — THE gate before any RL training run (spec Gates-1; the
round-trip descendant of probe_flip_scan.py, mandated by the 2026-07-05 pivot handoff).

From the floor-walk baseline: single-action counterfactuals (each decision span, each legal
alternative action, capped) -> full cached round trips -> per-probe realized-F1 deltas.
PASS = reward responds in BOTH directions with magnitude above the quantization step.
A support desert is a FINDING about the environment — report it, never work around it.

Only `scan_verdict` — the gate rule itself — is live; see main() for the retired runner.
"""


def scan_verdict(rows, mean_probes: float) -> dict:
    """PASS iff swaps moved realized recall BOTH directions with a quantization-exceeding
    magnitude in EACH direction: at least one up-move >= step AND one down-move <= -step
    (step = 1/mean probes per doc). A one-sided desert or sub-step wiggle FAILs."""
    step = 1.0 / max(mean_probes, 1.0)
    n_up = sum(1 for r in rows if r["delta"] > 0)
    n_down = sum(1 for r in rows if r["delta"] < 0)
    n_up_sig = sum(1 for r in rows if r["delta"] >= step)
    n_down_sig = sum(1 for r in rows if r["delta"] <= -step)
    max_abs = max((abs(r["delta"]) for r in rows), default=0.0)
    ok = n_up_sig >= 1 and n_down_sig >= 1
    return {"n_swaps": len(rows), "n_up": n_up, "n_down": n_down,
            "n_up_sig": n_up_sig, "n_down_sig": n_down_sig,
            "max_abs_delta": round(max_abs, 4), "quant_step": round(step, 4),
            "verdict": "PASS" if ok else "FAIL"}


def main():
    """RETIRED runner. The scan needed the ranker-v1 arms world
    (`train_ranker.{assemble,derive_spans,floor_walk_choice}`) and
    `roundtrip.roundtrip_batch`, both retired in commit e960fee ("refactor: retire legacy
    ranker-v1 stack"). `scan_verdict` above is unchanged and still the gate rule; the v2
    support scan that feeds it lives in `scripts/run_ranker_preflight.py`.
    """
    raise SystemExit(
        "retired runner: needs train_ranker + roundtrip_batch — see "
        "scripts/run_ranker_preflight.py; scan_verdict() in this file is still live"
    )


if __name__ == "__main__":
    main()
