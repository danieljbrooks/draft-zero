#!/usr/bin/env bash
# docs/031 E2: targets, on stage 0's first 40k games, at width W (default 128).
#   W=128 nohup caffeinate -i bash gorge/recipe/e2.sh > data/gorge/runs/r31/e2.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; W=${W:-128}
lim=$($PY gorge/recipe/limit.py $R/g0/pack 40000)
arm() { local name=$1; shift; echo "== $name $(date)"; train $R/e2/$name $W 7200 --train $R/g0/pack --limit-records $lim \
  --epochs 8 --patience 6 "$@"; }
arm visits-result  --policy-target visits --value-blend 0
arm visits-blend5  --policy-target visits --value-blend 0.5
arm cq03-result    --policy-target cq --cq-c-scale 0.03 --value-blend 0
arm cq03-blend3    --policy-target cq --cq-c-scale 0.03 --value-blend 0.3
arm cq10-blend5    --policy-target cq --cq-c-scale 0.1 --value-blend 0.5
arm cq01-blend5    --policy-target cq --cq-c-scale 0.01 --value-blend 0.5
echo "E2_DONE $(date)"
