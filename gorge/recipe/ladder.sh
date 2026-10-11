#!/usr/bin/env bash
# docs/031's search ladder, one level: a network inside the full search at SIMS simulations against gorge's bot (the
# fixed baseline), beside the search without a network at the same simulations. Each level plays at most PAIRS deck
# pairs, fewer if the last level's speed says they would take over MAX_H hours; under MIN_PAIRS it is skipped.
#   NET=<dir with best.dzgw> NAME=start SIMS=500 bash gorge/recipe/ladder.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; NAME=${NAME:?}; NET=${NET:?}; n=${SIMS:?}
PAIRS=${PAIRS:-500}; MAX_H=${MAX_H:-4}; MIN_PAIRS=${MIN_PAIRS:-50}
mkdir -p $R/ladder
rate=$R/ladder/$NAME.rate          # seconds per deck pair per simulation, both arms together, from the last level
pairs=$PAIRS
if [ -f $rate ]; then
  pairs=$(awk -v r="$(cat $rate)" -v n=$n -v h=$MAX_H -v p=$PAIRS 'BEGIN { q = int(h * 3600 / (r * n)); print (q < p ? q : p) }')
fi
if [ "$pairs" -lt "$MIN_PAIRS" ]; then
  echo "level $n: only $pairs pairs fit in $MAX_H h; skipped $(date)"; exit 0
fi
t0=$(date +%s)
for arm in net nonet; do
  out=$R/ladder/$NAME-$n; net=$NET/best.dzgw
  [ $arm = nonet ] && { out=$R/ladder/nonet-$n; net=""; }
  ta=$(date +%s)
  MATCH_SECS=$(( MAX_H * 3600 + 3600 )) vsbot "$net" "$out" $n $pairs
  echo "$arm@$n vs bot: $(score $out.jsonl) in $(( ($(date +%s) - ta) / 60 )) min $(date)"
done
awk -v s=$(( $(date +%s) - t0 )) -v p=$pairs -v n=$n 'BEGIN { print s / (p * n) }' > $rate
