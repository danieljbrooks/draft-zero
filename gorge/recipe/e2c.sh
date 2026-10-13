#!/usr/bin/env bash
# docs/031 E2c: regularising the value (it overfits after ~1 pass): width 128, visits + result, 40k games;
# then 80k games (stage 0 + its extension) at widths 64/128/256
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31
lim=$($PY gorge/recipe/limit.py $R/g0/pack 40000)
reg() { local name=$1; shift; echo "== $name $(date)"; train $R/e2/$name 128 7200 --train $R/g0/pack --limit-records $lim \
  --epochs 8 --patience 6 --policy-target visits --value-blend 0 "$@"; }
reg w128-vr-wd1    --wd 0.1
reg w128-vr-lr3e4  --lr 3e-4
reg w128-vr-drop2  --config "$(mlp_cfg 128 | sed 's/}$/,"dropout":0.2}/')"
until grep -q G0X_DONE $R/g0x.log 2>/dev/null; do sleep 60; done
for w in 64 128 256; do
  echo "== w$w-vr-g80k $(date)"
  train $R/e1/w$w-vr-g80k $w 7200 --train $R/g0/pack $R/g0x/pack --epochs 8 --patience 6 --policy-target visits --value-blend 0
done
echo "E2C_DONE $(date)"
