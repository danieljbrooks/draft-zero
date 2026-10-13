#!/usr/bin/env bash
# docs/031 E2b: the result-only value target at other widths (40k games)
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31
lim=$($PY gorge/recipe/limit.py $R/g0/pack 40000)
for w in 64 256; do
  for t in visits cq03; do
    tg=(--policy-target visits); [ $t = cq03 ] && tg=(--policy-target cq --cq-c-scale 0.03)
    echo "== w$w $t-result $(date)"
    train $R/e2/w$w-$t-result $w 7200 --train $R/g0/pack --limit-records $lim --epochs 8 --patience 6 "${tg[@]}" --value-blend 0
  done
done
echo "E2B_DONE $(date)"
