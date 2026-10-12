#!/usr/bin/env bash
# docs/031: the search's exploration constant at test time: NET inside the full search against gorge's bot.
#   NET=<best.dzgw> OUT=<dir> SPECS="0.25@100 0.1@100 0.5@500 1.5@500" PAIRS=500 bash gorge/recipe/cpuct.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
NET=${NET:?}; OUT=${OUT:?}; mkdir -p $OUT
for s in ${SPECS:?}; do
  c=${s%@*}; n=${s#*@}; p=${PAIRS:-500}; [ $n -gt 100 ] && p=${PAIRS_HI:-150}
  match $OUT/cpuct$c-$n.jsonl $n $p "az:sims=$n:$FULL:topk=6:cpuct=$c:gonet=$NET" bot
  echo "cpuct $c at $n: $(score $OUT/cpuct$c-$n.jsonl) $(date)"
done
echo CPUCT_DONE
