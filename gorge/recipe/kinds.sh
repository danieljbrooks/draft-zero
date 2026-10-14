#!/usr/bin/env bash
# docs/031 S2: which decision types does search pay for? The full search searches only the decisions of one kind (gorge's
# bot answers the rest), with the network NET or without one, at SIMS simulations against the bot (PAIRS deck pairs).
#   NET=<best.dzgw> SIMS=100 PAIRS=500 OUT=data/gorge/runs/r31/kinds bash gorge/recipe/kinds.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
NET=${NET:?}; SIMS=${SIMS:-100}; PAIRS=${PAIRS:-500}; OUT=${OUT:?}
mkdir -p $OUT
base="az:sims=$SIMS:autopay:oppnodes:oppfull${SEARCH_EXTRA:-}"   # SEARCH_EXTRA e.g. ":cpuct=0.25"
for kinds in all priority attackers blockers target modes,optional,choose,multitarget; do
  tag=${kinds%%,*}; [ "$kinds" = modes,optional,choose,multitarget ] && tag=rest
  for arm in net nonet; do
    pol="$base:kinds=$kinds"; [ $arm = net ] && pol="$pol:topk=6:gonet=$NET"
    match $OUT/$tag-$arm-$SIMS.jsonl $SIMS $PAIRS "$pol" bot
    echo "$tag $arm @$SIMS vs bot: $(score $OUT/$tag-$arm-$SIMS.jsonl) $(date)"
  done
done
echo "KINDS_DONE $SIMS $(date)"
