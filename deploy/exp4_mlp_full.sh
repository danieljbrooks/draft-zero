#!/bin/bash
# Experiment #4: the full-data MLP (configs/exp4_train_mlp.yml; docs/018), on one GPU of the second machine.
# Idempotent and resumable: a run in runs/exp4/mlp_full with latest.pt resumes; otherwise it waits for any MLP sweep
# still holding memory to exit (all the data needs most of the box's 40 GB), then starts. After training it scores the
# best policy and value checkpoints on the test split (test_*.json, once) and the best policy's rare decisions (val).
#   nohup deploy/exp4_mlp_full.sh [GPU] [CONFIG] [OUT] >> runs/exp4/mlp_full.log 2>&1 < /dev/null &
set -u
GPU=${1:-0}
CONFIG=${2:-configs/exp4_train_mlp.yml}
cd "$(dirname "$0")/.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
OUT=${3:-runs/exp4/mlp_full}
TABLES=data/imitation_scale/h5
log() { echo "[$(date -u +%H:%M:%S)] mlp_full: $*"; }

if [ -f "$OUT/summary.json" ]; then
  log "$OUT has finished training"
elif [ -f "$OUT/latest.pt" ]; then
  log "resuming $OUT"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --out "$OUT" --resume --tables-dir "$TABLES" || exit 1
else
  until ! pgrep -f "[s]weep --spec configs/exp4_sweep" > /dev/null; do log "waiting for the MLP sweeps to exit"; sleep 60; done
  log "training: $CONFIG"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --config "$CONFIG" --out "$OUT" \
      --tables-dir "$TABLES" || exit 1
fi

for c in best_policy best_value; do
  if [ -f "$OUT/$c.pt.gz" ] && [ ! -f "$OUT/test_$c.json" ]; then
    log "test split: $c"
    CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised evaluate --checkpoint "$OUT/$c.pt.gz" --split test \
        --tables-dir "$TABLES" --json "$OUT/test_$c.json"
  fi
done
if [ -f "$OUT/best_policy.pt.gz" ] && [ ! -f runs/exp4/rare_eval/$(basename "$OUT")-best_policy.json ]; then
  log "rare decisions (val): best_policy"
  CUDA_VISIBLE_DEVICES=$GPU python tools/imitation_scale/rare_eval.py "$OUT/best_policy.pt.gz" --out runs/exp4/rare_eval --rows 20000
fi
log "done"
