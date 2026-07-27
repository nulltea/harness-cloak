"""RETIRED: how many train probes CAN flip under a single-action counterfactual.

Needs the probe-era reward `u_qa` (src/cloak/train/reward.py) and the ranker-v1 arms world
(`train_ranker.{assemble,derive_spans}`), both retired in commit e960fee ("refactor:
retire legacy ranker-v1 stack"); recover them with `git show e960fee^:src/cloak/train/reward.py`.
The live successor on the v2 environment is the support scan in
`scripts/run_ranker_preflight.py` (round-trip reward, cached counterfactuals).
"""
raise SystemExit(
    "retired: needs the probe-era u_qa + train_ranker — see "
    "scripts/run_ranker_preflight.py"
)
