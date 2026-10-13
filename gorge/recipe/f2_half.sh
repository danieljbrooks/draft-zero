#!/usr/bin/env bash
# loop F' (loopf2.sh), on d1: the first half of generation K's self-play, deck pairs 0..HALF-1, resumable; DONE when finished.
#   bash gorge/recipe/f2_half.sh K SEED SIMS HALF
set -uo pipefail
cd "$(dirname "$0")/../.."
export PY=${PY:-/work/venv/bin/python} WORKERS=${WORKERS:-6} RECORD_EVERY=1
source gorge/recipe/lib.sh
k=$1; seed=$2; sims=$3; half=$4
R=data/gorge/runs/r31; o=$R/loops/F2/gen$k/d1; mkdir -p $o
p="az:sims=$sims:explore:$FULL:cpuct=0.25:topk=6:gonet=$R/loops/F2/gen$((k - 1))/best.dzgw"
s=$(cat $o/next 2>/dev/null || echo 0)
while [ $s -lt $half ]; do
  e=$(( s + 100 < half ? s + 100 : half ))
  m=$(_chunk $o $s $((e - s)) $seed "$p" 7200)
  s=$((s + m)); [ $s -lt $e ] && s=$((s + 1))
  echo $s > $o/next
done
touch $o/DONE
