#!/usr/bin/env bash
# docs/031 loop F: fewer, better games. Each generation: GAMES games of self-play at SIMS simulations (cpuct 0.25, every
# decision recorded) with the previous network; a warm start on the last WINDOW generations' games with the search's
# root value as the value's label (VB); a head-to-head against the fixed reference REF (cpuct 0.25, 100 simulations,
# 500 deck pairs). Generation 1 is E7's root-value network (its self-play: E7's games). Resumable.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; L=$R/loops/F; REF=${REF:?reference best.dzgw}
SIMS=${SIMS:-200}; GAMES=${GAMES:-1200}; WINDOW=${WINDOW:-3}; VB=${VB:-1}
mkdir -p $L/gen1
[ -e $L/gen1/sp ] || ln -s ../../../e7/sp $L/gen1/sp
for f in best.pt best.dzgw config.json; do [ -e $L/gen1/$f ] || cp $R/e7/vb1/$f $L/gen1/; done
for ((k = ${FIRST:-2}; k <= ${MAXGEN:-100}; k++)); do
  [ -f $R/STOP_F ] && { echo "STOP_F"; break; }
  prev=$L/gen$((k - 1)); cur=$L/gen$k
  echo "== F gen $k: self-play $(date)"
  RECORD_EVERY=1 SP_EXTRA=":cpuct=0.25" CHUNK=100 CHUNK_SECS=7200 selfplay $cur/sp $GAMES $((7700 + k)) $prev/best.dzgw $SIMS
  echo "   $(wc -l < $cur/sp/games.jsonl) games $(date)"
  packs=(); for ((j = k - WINDOW + 1; j <= k; j++)); do [ $j -ge 1 ] && packs+=($L/gen$j/sp/pack); done
  train $cur 128 14400 --train "${packs[@]}" --init $prev/best.pt --epochs 2 --lr 1e-4 --warmup 100 --patience 4 \
    --policy-target visits --value-blend $VB
  match $cur/vsref.jsonl 100 500 "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$cur/best.dzgw" \
    "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$REF"
  echo "   F$k vs reference: $(score $cur/vsref.jsonl) $(date)"
done
echo LOOPF_DONE
