#!/usr/bin/env bash
# docs/031 E3: one self-play loop, generation by generation, resumable (finished steps are skipped).
#
#   NAME=A WIDTH=64 MODE=scratch SIMS=25 GAMES=10000 GENS=3 START=<gen-0 dir with best.dzgw> bash gorge/recipe/loop.sh
#
# Generation k (k >= 1): cheap self-play with generation k-1's network in the search, then the next network trains
# on the window (BASE, stage 0's games, plus every generation's so far; at most WINDOW packs), then the strength
# match against the zero point and, if H2H > 0, the head-to-head against generation k-1.
#   MODE=scratch  a fresh network each generation (docs/027's loop)
#   MODE=warm     generation k-1's weights, WARM_EPOCHS passes at WARM_LR
set -uo pipefail
source "$(dirname "$0")/lib.sh"
NAME=${NAME:?}; WIDTH=${WIDTH:-64}; MODE=${MODE:-scratch}; SIMS=${SIMS:-25}; GAMES=${GAMES:-10000}
GENS=${GENS:-3}; START=${START:?START=gen-0 network dir}; FIRST=${FIRST:-1}; WINDOW=${WINDOW:-20}
WARM_EPOCHS=${WARM_EPOCHS:-1}; WARM_LR=${WARM_LR:-1e-4}; LR=${LR:-3e-4}; PAIRS=${PAIRS:-400}; H2H=${H2H:-0}
TARGETS=(${TARGETS:---policy-target visits --value-blend 0})
BASE=(${BASE:-data/gorge/runs/r31/g0/pack data/gorge/runs/r31/g0x/pack})
R=data/gorge/runs/r31; L=$R/loops/$NAME
mkdir -p $L
[ -e $L/gen0 ] || ln -s "$(cd "$START" && pwd)" $L/gen0
for ((k = FIRST; k < FIRST + GENS; k++)); do
  prev=$L/gen$((k - 1)); cur=$L/gen$k
  echo "== $NAME gen $k: self-play $(date)"
  selfplay $cur/sp $GAMES $((3200 + 100 * k + $(printf %d "'${NAME:0:1}"))) $prev/best.dzgw $SIMS
  echo "   $(wc -l < $cur/sp/games.jsonl) games $(date)"
  packs=("${BASE[@]}")
  for ((j = 1; j <= k; j++)); do packs+=($L/gen$j/sp/pack); done
  packs=("${packs[@]: -$WINDOW}")
  echo "   train on ${#packs[@]} packs $(date)"
  if [ "$MODE" = warm ]; then
    train $cur $WIDTH 14400 --train "${packs[@]}" --init $prev/best.pt --epochs $WARM_EPOCHS --lr $WARM_LR \
      --warmup 100 --patience 4 "${TARGETS[@]}"
  else
    train $cur $WIDTH 14400 --train "${packs[@]}" --epochs 8 --patience 6 --lr $LR "${TARGETS[@]}"
  fi
  echo "   strength $(date)"
  [ -f $cur/strength.jsonl ] || strength $cur/best.dzgw $cur/strength $PAIRS
  echo "   strength: $(score $cur/strength.jsonl)"
  if [ "$H2H" -gt 0 ]; then
    [ -f $cur/h2h.jsonl ] || h2h $cur/best.dzgw $prev/best.dzgw $cur/h2h $H2H
    echo "   vs gen $((k - 1)): $(score $cur/h2h.jsonl) $(date)"
  fi
done
echo "LOOP_DONE $NAME $(date)"
