#!/usr/bin/env bash
# docs/031's search ladder: a network inside the full search at more and more simulations against gorge's bot (the
# fixed baseline), beside the search without a network at the same simulations. Doubles the simulations until a
# match would take longer than MAX_H hours (from the last match's time) or the level SIMS_MAX.
#   NET=<dir with best.dzgw> NAME=gen0 bash gorge/recipe/ladder.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; NAME=${NAME:?}; NET=${NET:?}; PAIRS=${PAIRS:-500}; MAX_H=${MAX_H:-4}; SIMS_MAX=${SIMS_MAX:-3200}
mkdir -p $R/ladder
for ((n = ${SIMS_MIN:-25}; n <= SIMS_MAX; n *= 2)); do
  for arm in net nonet; do
    out=$R/ladder/$NAME-$n; net=$NET/best.dzgw
    [ $arm = nonet ] && { out=$R/ladder/nonet-$n; net=""; }
    t0=$(date +%s)
    vsbot "$net" "$out" $n $PAIRS
    secs=$(( $(date +%s) - t0 ))
    echo "$arm@$n vs bot: $(score $out.jsonl) in $(( secs / 60 )) min $(date)"
    [ $arm = net ] && last=$secs
  done
  if [ $(( 2 * last )) -gt $(( MAX_H * 3600 )) ]; then echo "next level would take over $MAX_H h; stop"; break; fi
  [ -f $R/STOP_LADDER ] && { echo "STOP_LADDER"; break; }
done
echo "LADDER_DONE $NAME $(date)"
