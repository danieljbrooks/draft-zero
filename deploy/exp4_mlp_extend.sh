#!/bin/bash
# Experiment #4: extend the full-data MLP (run 2, runs/exp4/mlp_1ep) one epoch at a time, towards 20 in total (Dan,
# Sat 3 October; docs/018, "Extending run 2"), on one GPU of the second machine. Epoch k is runs/exp4/mlp_ext<k>, a new
# run from epoch k-1's final weights (configs/exp4_train_mlp_ext.yml with wave K's peak rate and decay).
# tools/imitation_scale/extend_rule.py decides after each epoch; at the stop the best epoch's best_policy is scored on
# the test split (once) and on rare decisions. Idempotent and resumable: rerun it and it carries on.
#   nohup deploy/exp4_mlp_extend.sh GPU LR WD [MAX_EPOCHS] >> runs/exp4/mlp_ext.log 2>&1 < /dev/null &
set -u
GPU=$1 LR=$2 WD=$3 MAX=${4:-20}
cd "$(dirname "$0")/.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
TABLES=data/imitation_scale/h5
CONFIG=configs/exp4_train_mlp_ext.yml
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] mlp_ext: $*"; }

until ! pgrep -f "[s]weep --spec configs/exp4_sweep" > /dev/null; do log "waiting for the MLP sweeps to exit"; sleep 60; done
RUNS=(runs/exp4/mlp_1ep)
k=2
while :; do
  DECISION=$(python tools/imitation_scale/extend_rule.py "${RUNS[@]}" --max-epochs "$MAX") || exit 1
  log "after epoch $((k - 1)): $DECISION"
  [ "${DECISION%% *}" = continue ] || break
  OUT=runs/exp4/mlp_ext$k
  PREV=${RUNS[-1]}/final.pt.gz
  if [ -f "$OUT/summary.json" ]; then
    log "epoch $k already trained"
  elif [ -f "$OUT/latest.pt" ]; then
    log "epoch $k: resuming"
    CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --out "$OUT" --resume --tables-dir "$TABLES" \
        || exit 1
  else
    log "epoch $k: from $PREV, peak lr $LR, weight decay $WD"
    CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --config "$CONFIG" --out "$OUT" \
        --tables-dir "$TABLES" --set "init_checkpoint=$PREV" --set "lr=$LR" --set "weight_decay=$WD" || exit 1
  fi
  RUNS+=("$OUT")
  k=$((k + 1))
done

BEST=${DECISION#* }
log "best: $BEST"
if [ ! -f "$BEST/test_best_policy.json" ]; then
  log "test split: $BEST/best_policy"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised evaluate --checkpoint "$BEST/best_policy.pt.gz" \
      --split test --tables-dir "$TABLES" --json "$BEST/test_best_policy.json"
fi
if [ ! -f "runs/exp4/rare_eval/$(basename "$BEST")-best_policy.json" ]; then
  log "rare decisions (val): $BEST/best_policy"
  CUDA_VISIBLE_DEVICES=$GPU python tools/imitation_scale/rare_eval.py "$BEST/best_policy.pt.gz" --out runs/exp4/rare_eval \
      --rows 20000
fi
echo "$BEST" > runs/exp4/mlp_ext_best.txt
log "done"
