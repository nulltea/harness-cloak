#!/usr/bin/env bash
# Driver for the base ORG-safe generality-mix experiment (spec: docs/specs/detector-model.md;
# plan: research-wiki/training/2026-07-05-FT-detector-v5-base-orgsafe-genmix.md).
#
# Builds + trains the staged Pile-share sweep on the BASE backbone at max_width=60:
#   - one build+train per PILE_FRAC arg (default: just 0.18, the point of interest)
#   - one width-control train on the EXISTING v2 dataset (Pile 10%) at max_width=60
# v2 (Pile 10%) and v4 (Pile 25%) are already run — this fills the interior point(s).
# Does NOT run eval (the dev sweep / generality probe are separate, slower steps — see NEXT below).
#
# Usage:  bash scripts/spikes/detector_orgsafe_genmix_sweep.sh [PILE_FRAC ...]
#   Stage 1 (default):  bash .../detector_orgsafe_genmix_sweep.sh            # -> 0.18 + control
#   Stage 2 (if needed): bash .../detector_orgsafe_genmix_sweep.sh 0.15 0.22 # add refinement points
#
# GPU: one process at a time (gfx1151). Runs serial. Unbuffered logs -> results/.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="PYTHONPATH=src .venv/bin/python -u"
FRACS=("${@:-0.18}")                      # default single point 0.18
MIX="nemotron=30000,pilener=40000,wikibio=corpora/wikipedia_bio/train.json"
BASE="knowledgator/gliner-pii-base-v1.0"
# NOTE: max_width stays NATIVE (12). Widening to 60 is infeasible on the pretrained span head
# (Linear(hidden, hidden*max_width) is baked at the trained width — see docs/specs/detector-model.md C5),
# so there is no --max-width and no width-control run. v5 = single-variable-from-v2 (mix only).
# v2 recipe: 3 epochs, lr 1e-5 / others 5e-5, batch 8, seed 42.
COMMON="--epochs 3 --batch-size 8 --lr 1e-5 --others-lr 5e-5 --seed 42"

for pf in "${FRACS[@]}"; do
  tag="p$(echo "$pf" | tr -d '.')"        # 0.18 -> p018
  data="data/pii_span_dataset_orgsafe_$tag"
  out="data/models/pii_gliner_orgsafe_$tag"
  echo "===== BUILD pile-frac=$pf -> $data ====="
  eval $PY scripts/build_pii_span_dataset.py --mix "$MIX" --balance-rare --pile-frac "$pf" \
    --out-dir "$data" 2>&1 | tee "results/build_orgsafe_$tag.log"
  echo "===== TRAIN $BASE @max_width=$MW -> $out ====="
  eval $PY scripts/train_pii_gliner.py --init "$BASE" --data-dir "$data" $COMMON --out "$out" \
    2>&1 | tee "results/train_orgsafe_$tag.log"
done

cat <<'NEXT'
== NEXT (eval — not run here; separate GPU steps) ==
For each data/models/pii_gliner_orgsafe_*/checkpoint-*:
  dev sweep:   scripts/latticecloak_detection_gate.py --corpus corpora/tab/echr_dev.json --threshold T --gliner-model <ckpt>
               (sweep T in 0.02 0.05 0.1 0.2 0.3; select by recall-at-matched-precision; reject dev QUASI/ORG < 0.90)
  TAB test:    scripts/latticecloak_detection_gate.py --corpus corpora/tab/echr_test.json --threshold 0.02 --gliner-model <sel>
  generality:  scripts/spikes/pii_zeroshot_generality.py --gliner-model <sel> --threshold 0.3
Read ORG recall vs build_shares.json TAB/ORG share across pile fracs (+ v2/v4) to test the dilution hypothesis.
NEXT
