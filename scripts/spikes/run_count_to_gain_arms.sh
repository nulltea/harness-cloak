#!/usr/bin/env bash
# Count-to-gain coupling arms — paired seed, one GPU process at a time.
# Record: research-wiki/experiments/2026-08-03-count-to-gain-coupling.md
#
# Each arm gets its OWN cache copy: the arms visit different action vectors, so a
# shared cache would let the first arm write entries the second reads, making the
# paired comparison arm-order dependent. --cache-only is NOT used (it aborts on
# the first miss, and an RL run necessarily misses on fresh rollouts).
set -euo pipefail
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# The LLM cache is SHARED on purpose. It is content-addressed by prompt, so an
# identical prompt returns an identical response regardless of which arm asked
# first — sharing it cannot make the comparison arm-order dependent, and
# isolating it would force full regeneration at real remote cost. The utility
# cache is different: it is keyed by action vector and ACCUMULATES rows as arms
# explore, so it is copied per arm above.
export CLOAK_LLM_CACHE=data/llm_cache

SEED="${1:?usage: run_count_to_gain_arms.sh <seed> [arm ...]}"
shift || true
# Arms default to both; pass one to resume a pair (e.g. "coupled") after the
# other has already completed.
ARMS="${*:-detached coupled}"
OUT=results/ranker_v2/architecture/count_to_gain
BASE=results/ranker_v2
PROD=$BASE/architecture/controller_production

for ARM in $ARMS; do
  echo "=== count-to-gain $ARM seed $SEED ($(date -Is)) ==="
  .venv/bin/python -u scripts/train_interactive_ranker.py train \
    --policy-architecture semantic-v1 \
    --environment $BASE/environment/ranker-env.json \
    --representation-manifest $BASE/architecture/representation-full/manifest.json \
    --profile-count-targets $BASE/reward/profile-count-targets.json \
    --utility-artifact $BASE/qa/aci-full.utility \
    --utility-cache "$OUT/cache-$ARM-s$SEED.jsonl" \
    --threshold-manifest $BASE/preflight/threshold-manifest.json \
    --lambda-menu $BASE/preflight/lambda-menu.json \
    --exit-winners "$PROD/exit-winners-s$SEED.json" \
    --bc-checkpoint "$PROD/bc-s$SEED.pt" \
    --out-checkpoint "$OUT/$ARM-s$SEED.pt" \
    --kl-reference-checkpoint "$OUT/kl-ref-$ARM-s$SEED.pt" \
    --epoch-reports "$OUT/epochs-$ARM-s$SEED.jsonl" \
    --fixed-lambda-zero-control "$OUT/lambda-zero-$ARM-s$SEED.json" \
    --doc-id aci/D2N005 --doc-id aci/D2N027 \
    --doc-id aci/D2N031 --doc-id aci/D2N063 \
    --seed "$SEED" --device auto \
    --remote-workers 6 --reader-workers 6 \
    --max-epochs 8 --rollouts 8 \
    --learning-rate 1e-4 --beta 0.01 --eta 0.01 \
    --alpha-utility-routing none --controller-gap-scaling none \
    --alpha-init switch-calibrated --rollout-scaling fixed \
    --counterfactual-coverage degeneracy \
    --kl-schedule collapse-trigger --kl-direction forward \
    --synchronous-profile-eval --synchronous-profile-samples 16 \
    --utility-logit-softcap 25 --profile-sensitivity-reg 0.1 \
    --controller-gain evidence --controller-gain-hidden 32 \
    --controller-gain-lr 1e-2 --controller-gain-bound 1.5 \
    --tie-mode online --tie-coefficient 1.0 --tie-margin 0.1 \
    --tie-min-contexts 3 --gain-penalty 1e-3 \
    --tie-evidence-bootstrap --batched-rollouts \
    --count-to-gain "$ARM" \
    2>&1 | tee "$OUT/$ARM-s$SEED.log"
done
