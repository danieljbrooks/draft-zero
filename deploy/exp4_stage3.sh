#!/bin/bash
# Experiment #4, stage 3 (docs/018): the large training of the transformer, on one GPU of a machine that has the repo,
# the venv (python on PATH) and the tables in data/imitation_scale/h5. Idempotent and resumable:
#   - a run already in runs/exp4/main (latest.pt) resumes;
#   - otherwise it waits for the two 30% checks, settles the two open settings from them
#     (tools/imitation_scale/stage3_choice.py), stops the scale-check sweeps to free their memory, and starts.
# Then it scores the best policy and value checkpoints on the test split (runs/exp4/main/test_*.json).
#
#   nohup deploy/exp4_stage3.sh 0 >> runs/exp4/main.log 2>&1 < /dev/null &      # GPU 0
set -u
GPU=${1:-0}
cd "$(dirname "$0")/.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
OUT=runs/exp4/main
TABLES=data/imitation_scale/h5
log() { echo "[$(date -u +%H:%M:%S)] stage3: $*"; }

if [ -f "$OUT/summary.json" ]; then
  log "$OUT has finished training"
elif [ -f "$OUT/latest.pt" ]; then
  log "resuming $OUT"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --out "$OUT" --resume --tables-dir "$TABLES" || exit 1
else
  log "waiting for the 30% checks"
  until [ -f runs/exp4/sweep_s30/runs/s30-l1-actor3-td99/summary.json ] && \
        [ -f runs/exp4/sweep_s30b/runs/s30-l1-actor3-vw02/summary.json ]; do sleep 60; done
  mapfile -t SETS < <(python tools/imitation_scale/stage3_choice.py --runs runs/exp4 --out "$OUT/choice.json")
  python -c "import json; [print('  ' + w) for w in json.load(open('$OUT/choice.json'))['why']]"
  # the scale-check sweeps hold ~10 GB each: stop them (their runs are done) before loading all the data
  for s in sweep_s30 sweep_s30b; do touch "runs/exp4/$s/STOP"; done
  pkill -TERM -f "[s]weep --spec configs/exp4_sweep_s30" ; sleep 30
  pkill -KILL -f "[s]weep --spec configs/exp4_sweep_s30" ; sleep 5
  ARGS=()
  for s in "${SETS[@]}"; do ARGS+=(--set "$s"); done
  log "training: configs/exp4_train.yml ${SETS[*]:-}"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised train --config configs/exp4_train.yml --out "$OUT" \
      --tables-dir "$TABLES" "${ARGS[@]}" || exit 1
fi

for c in best_policy best_value; do
  if [ -f "$OUT/$c.pt.gz" ] && [ ! -f "$OUT/test_$c.json" ]; then
    log "test split: $c"
    CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised evaluate --checkpoint "$OUT/$c.pt.gz" --split test \
        --tables-dir "$TABLES" --json "$OUT/test_$c.json"
  fi
done
log "done"
