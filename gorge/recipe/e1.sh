#!/usr/bin/env bash
# docs/031 E1: MLP width x games, on stage 0's games (r31/g0), docs/027 loop B's targets.
#   nohup caffeinate -i bash gorge/recipe/e1.sh > data/gorge/runs/r31/e1.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31
until grep -q G0_DONE $R/g0.log 2>/dev/null; do sleep 60; done
read -r L5 L10 L20 L40 < <($PY gorge/recipe/limit.py $R/g0/pack 5000 10000 20000 40000 | tr '\n' ' ')
echo "records: 5k $L5, 10k $L10, 20k $L20, 40k $L40"
for n in 5 10 20 40; do
  eval "lim=\$L$n"
  for w in 64 128 256 384; do
    echo "== w$w g${n}k $(date)"
    train $R/e1/w$w-g${n}k $w 7200 --train $R/g0/pack --limit-records "$lim" --epochs 8 --patience 6 \
      --policy-target cq --cq-c-scale 0.03 --value-blend 0.5
  done
done
echo "E1_DONE $(date)"
