#!/usr/bin/env bash
# AlphaZero-style generations with a dzg network: self-play with the last network, train the next one
# on every generation's data so far (warm-started from the last), and play it against the search
# without a network and against the last generation.
#
#   LOOP=data/gorge/runs/loopA START=data/gorge/runs/g1/mlp/best.pt GENS="2 3" \
#     TRAIN_ARGS="--epochs 4 --patience 4" DATA="data/gorge/runs/g0/pack" bash gorge/dzg_loop.sh
#
# START is generation GENS[0]-1's network; DATA lists the packed corpora it was trained on (each
# generation's self-play is added to it). VAL is the held-out eval-deck pack. Every step is
# hard-killed: self-play after SP_MIN minutes, training after 120, evaluation after EV_MIN per arm.
set -euo pipefail
source "$(dirname "$0")/dzg_lib.sh"
LOOP="${LOOP:?LOOP=dir}"; START="${START:?START=checkpoint}"; GENS="${GENS:-2 3}"
DATA="${DATA:?DATA=packed corpora}"; VAL="${VAL:-$D/runs/g0/pack_eval}"
SIMS="${SIMS:-100}"; SP_PAIRS="${SP_PAIRS:-6000}"; SP_MIN="${SP_MIN:-75}"; W="${W:-60}"
EV_PAIRS="${EV_PAIRS:-300}"; EV_MIN="${EV_MIN:-45}"; TRAIN_ARGS="${TRAIN_ARGS:---epochs 4 --patience 4}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"; SEED0="${SEED0:-300}"
mkdir -p "$LOOP"
prev="$START"
for g in $GENS; do
  echo "=== generation $g: self-play with $prev ($(date -u +%H:%M) UTC)"
  OUT="$LOOP/g$g-selfplay" NET="$prev" SIMS="$SIMS" PAIRS="$SP_PAIRS" W="$W" SEED=$((SEED0 + g)) MAXMIN="$SP_MIN" \
    bash "$DZ/gorge/dzg_gen.sh"
  DATA="$DATA $LOOP/g$g-selfplay/pack"
  echo "=== generation $g: training on $DATA ($(date -u +%H:%M) UTC)"
  mkdir -p "$LOOP/g$g"
  # shellcheck disable=SC2086
  PYTHONPATH="$DZ/gorge" timeout 120m "$PY" -m dzg.train --init "$prev" --train $DATA --val "evaldecks=$VAL" \
    --out "$LOOP/g$g" --in-ram --device "$TRAIN_DEVICE" $TRAIN_ARGS > "$LOOP/g$g/train.log" 2>&1
  echo "=== generation $g: evaluation ($(date -u +%H:%M) UTC)"
  NET="$LOOP/g$g/best.pt" PREV="$prev" OUT="$LOOP/eval-g$g" SIMS="$SIMS" PAIRS="$EV_PAIRS" W="$W" \
    ARMS="full prev" MAXMIN="$EV_MIN" bash "$DZ/gorge/dzg_eval.sh"
  prev="$LOOP/g$g/best.pt"
done
echo "=== loop done ($(date -u +%H:%M) UTC)"
